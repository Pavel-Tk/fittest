# ALCHEMI Toolkit 0.3.0 (rc) API notes for TrustSwarm

Source: `ref/nvalchemi-toolkit` (package `nvalchemi`, `__version__ = "0.3.0"`, `nvalchemi/__init__.py:18`).
All paths below are relative to that repo. `file:line` points at the definition or the example line.
Everything marked **UNVERIFIED** is not in the repo (e.g. Orb APIs) and must be checked against the external package.

Units everywhere: positions Å, energy eV (per system, shape `[B,1]`), forces eV/Å (per atom, `[V,3]`),
stress eV/Å^3, FIRE/MD `dt` in fs, masses amu. (`.claude/skills/nvalchemi-data-structures/SKILL.md` field table, `nvalchemi/models/dftd3.py:439`.)

---

## 0. TL;DR design for TrustSwarm (what the source supports)

* In-flight batching exists only through `FusedStage.run(batch=None)` (Mode 2) with a `SizeAwareSampler` (`nvalchemi/dynamics/base.py:4982-5085`). A plain `FIRE2(...).run(batch)` never refills (`base.py:3243`).
* A system leaves the live batch when `batch.status >= exit_status`. Any hook can set that, and the next `refill_check` evicts it and pulls a replacement (`base.py:3305-3445`). This is how DisagreementHook "graduates" a system. No special API is needed.
* There is no built-in "reason" tag. `status` is a free int and refill uses `>=`, so `exit_status+1` would work in Mode 2, but `all_complete()` in Mode 1 uses `==` (`base.py:4899`). **Recommended:** set `status = exit_status` and keep the reason in the hook's own Python dict keyed by `system_id`, which the sampler stamps (`sampler.py:334-339`).
* Dynamics takes ONE model. The other two committee members are called inside the hook (or wrap all 3 in a custom committee `BaseModelMixin`; extra outputs get published to the batch automatically, `base.py:2958`).
* MACE and AIMNet2 do NOT build their own neighbor lists. You must register `model.make_neighbor_hooks()`, or call `nvalchemi.neighbors.compute_neighbors(batch, config=model.model_config.neighbor_config)` before a direct `model(batch)`.
* The "acceptance report" is `nvalchemi.training.distillation.evaluation.build_acceptance_report`. It lives in the distillation package and is model-agnostic. Use `baseline_accuracy` + `max_from_scratch_ratio=1.0` to get a "fine-tuned beats pre-fine-tune" verdict.

---

## 1. Data structures

```python
import torch, numpy as np
from nvalchemi.data import AtomicData, Batch
```

`AtomicData` (`nvalchemi/data/atomic_data.py`) is a Pydantic model. Required fields are `positions [V,3]` and `atomic_numbers [V]`. System-level fields use shape `[1, ...]`: `energy [1,1]`, `charge [1,1]` (total charge, `atomic_data.py:336`), `cell [1,3,3]`, `pbc [1,3]`. Per-atom `charges [V]` are partial charges, a different field (`:330`).

### From numpy (non-periodic molecule)
```python
def to_atomic_data(z: np.ndarray, pos: np.ndarray, E=None, F=None, charge=0, device="cuda"):
    n = len(z)
    kw = dict(
        atomic_numbers=torch.as_tensor(z, dtype=torch.long),
        positions=torch.as_tensor(pos, dtype=torch.float32),
        charge=torch.tensor([[float(charge)]]),          # AIMNet2 requires 'charge'
        forces=torch.as_tensor(F, dtype=torch.float32) if F is not None else torch.zeros(n, 3),
        energy=torch.tensor([[float(E)]]) if E is not None else torch.zeros(1, 1),
    )
    d = AtomicData(**kw)                                  # cell/pbc left None => non-periodic
    d.add_node_property("velocities", torch.zeros(n, 3))  # FIRE2/MD update velocities in place
    return d.to(device)
```
* Leave `cell`/`pbc` as `None` for molecules. MACE fills an identity cell and zero shifts itself (`nvalchemi/models/mace.py:812-832`). `from_atoms` sets both to `None` when `atoms.pbc` is all False (`atomic_data.py:906-909`).
* `atomic_masses` is auto-populated from the periodic table. FIRE2 needs `velocities` + `atomic_masses` on the initial batch (`base.py:2140-2158` `required_input_keys`, FIRE2 `__provides_keys__={"positions","velocities"}` `optimizers/fire2.py:151`).

### From ASE (`atomic_data.py:843`)
```python
AtomicData.from_atoms(atoms, energy_key="energy", forces_key="forces", stress_key="stress",
                      virials_key="virials", dipole_key="dipole", charges_key="charges",
                      device="cpu", dtype=torch.float32, z_table=None)
```
* It reads `atoms.info[energy_key]` and `atoms.arrays[forces_key]` ONLY (`:925-938`). Calculator results are **not** read. Copy them first, as `examples/intermediate/08_lora_finetuning.py:192-204` does: `atoms.info["energy"]=E; atoms.arrays["forces"]=F`.
* Total charge comes from `atoms.info["charge"]` (must be an int) or from the sum of `charges` (`:974-1012`).
* Numeric `atoms.info` entries go to `data.info` (a dict of tensors). Strings are dropped.
* `atomic_numbers` comes out int32 here (`:900`). That is fine for the models.

### Batch (`nvalchemi/data/batch.py`)
```python
batch = Batch.from_data_list(list_of_atomic_data, device="cuda")   # batch.py:1160
batch.num_graphs          # B (batch.py:857); len(batch) same
batch.batch_idx           # [V] graph index per atom (batch.py:887)
batch.batch_ptr           # [B+1] atom offsets (batch.py:895)
batch.num_nodes_per_graph # [B] (batch.py:1069)
batch.energy  # [B,1]; batch.forces [V,3]; batch.positions [V,3]
batch[3]                  # -> AtomicData (get_data, batch.py:1563)
batch[[0, 2]] / batch[mask] / batch[1:3]  # -> Batch
sub = batch.index_select(idx_tensor, drop={"neighbor_list"})  # batch.py:1669, copy, no AtomicData rebuild
batch.to_data_list()      # batch.py:1656
batch.clone()
```
Per-system custom fields:
```python
batch["flag"] = torch.zeros(batch.num_graphs, 1, device=batch.device)   # [B,1] → system level (how status is stored, base.py:1788)
batch.add_key("t1x_idx", [torch.tensor([[i]]) for i in ids], level="system")  # batch.py:3040
data.add_system_property("t1x_idx", torch.tensor([[i]]))                 # per AtomicData, atomic_data.py:738
```
**Gotcha:** custom system keys are lost in a `to_data_list()` → `from_data_list()` round trip because `__system_keys__` is reset. This hits `HostMemory` sinks (`examples/intermediate/04_inflight_batching.py:87-111` subclasses `HostMemory` to keep `system_id`). Read custom fields from the live batch inside a hook instead of from sinks.

Per-system reductions:
```python
fmax = torch.zeros(B, device=d).scatter_reduce_(0, batch.batch_idx.long(),
            batch.forces.norm(dim=-1), reduce="amax", include_self=False)
```

---

## 2. Models

Contract: `model(batch) -> OrderedDict` (`ModelOutputs`) with `energy [B,1]` (eV) and `forces [V,3]` (eV/Å). Optional keys are `stress [B,3,3]`, `charges [V]`, and `atomic_energies [V]` (MACE).
Keys are the plain names (`"energy"`, `"forces"`). Select outputs with `model.set_config("active_outputs", {...})` or `model.model_config.active_outputs = {...}` (`nvalchemi/models/base.py:767`).

### MACE (`nvalchemi/models/mace.py:509`, `from_checkpoint` `:1034`)
```python
from nvalchemi.models.mace import MACEWrapper
mace = MACEWrapper.from_checkpoint(
    "medium-mpa-0",              # exact name; docstring mace.py:1067, LoRA example :124
    device=torch.device("cuda"),
    dtype=torch.float32,         # cast weights; None keeps checkpoint dtype
    enable_cueq=False,           # True needs cuequivariance + cuequivariance-ops-torch-cu12/13, CUDA only (:1135-1160)
    compile_model=False,
    atomic_energies=None,        # {Z: E0_eV} overrides (use for T1x re-referencing, see §8)
    train=False,                 # False freezes weights + eval(); True for fine-tuning
)
```
* Names are resolved by `mace.calculators.foundations_models.download_mace_mp_checkpoint` (`:1122`). The repo also uses `"small"`, `"medium"`, `"medium-0b2"`, and `"small-0b"`.
* Outputs: `{"energy","forces","stress","atomic_energies"}`. Default active: `{"energy","forces"}` (`:582-586`). Forces come from autograd.
* Needs a **COO** neighbor list: `neighbor_config=NeighborConfig(cutoff=model.r_max, format=COO, half_list=False)` (`:592-596`). `adapt_input` reads `data.neighbor_list` and fails without it (`:799`).

### AIMNet2 (`nvalchemi/models/aimnet2.py:164`, `from_checkpoint` `:277`)
```python
from nvalchemi.models.aimnet2 import AIMNet2Wrapper
aim = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cuda", compile_model=False)
```
* Checkpoint names are whatever `aimnet.calculators.AIMNet2Calculator(model=<str>)` resolves (`:314-322`). The repo uses `"aimnet2"` (`aimnet2.py:28`), `"aimnet2_wb97m_d3_3"` (`examples/advanced/08_aimnet2_ewald_pipeline.py:181`), and `"aimnet2-rxn"` (only through the calculator directly, `examples/advanced/11_batched_neb.py:227`). Other names depend on the installed `aimnet` registry (**UNVERIFIED**).
* **float32 only.** Weights are cast with `.float()` (`:327`). `forward` casts `data["positions"]` and `data["cell"]` to fp32 **in place on your batch** (`:656-664`), which matters if you pass a float64 batch.
* **Charge:** `required_inputs={"charge"}`. Missing charge silently defaults to 0 (`:501-507`). Always set `charge=[[0.]]` explicitly.
* **D3 and Coulomb are disabled** inside the wrapper (`:226-235`). Raw `AIMNet2Wrapper` is therefore NOT the full published AIMNet2 (the wB97M-D3 model). Add D3 through a pipeline (below) or use the calculator-based wrapper pattern from example 11 (§3).
* Needs a **MATRIX** neighbor list (`:262-268`). Outputs: energy, forces, stress, charges. Default active: `{"energy","forces","charges"}`.

### Calling a model directly (outside dynamics)
```python
from nvalchemi.neighbors import compute_neighbors        # nvalchemi/neighbors.py:140
def predict(model, batch):
    b = batch.clone()
    nc = model.model_config.neighbor_config
    if nc is not None:
        compute_neighbors(b, config=nc)                    # format/cutoff taken from config (COO for MACE, MATRIX for AIMNet2)
    with torch.enable_grad():                              # autograd forces need grad
        out = model(b)
    return out["energy"].detach(), out["forces"].detach()  # [B,1] eV, [V,3] eV/Å
```
The equivalent with a hook is used in the LoRA example (`08_lora_finetuning.py:292-298`): `NeighborListHook(nc, stage=...)(HookContext(batch=b, model=m), stage)`.
**Do not reuse one batch for two models with different neighbor configs.** `compute_neighbors` overwrites the batch's neighbor data. Clone the batch per model.

### PipelineModelWrapper (`nvalchemi/models/pipeline.py:234/279/319`)
* `PipelineStep(model, wire={out_key: batch_attr})`: rename outputs so they feed downstream inputs.
* `PipelineGroup(steps=[...], use_autograd=False, derivative_fn=None)`: with `use_autograd=True`, the members return energy only, the group sums them, and forces are taken as -d(sum E)/dr. With `False`, each member computes its own forces and the outputs are summed.
* `PipelineModelWrapper(groups, additive_keys={"energy","forces","stress"}, neighbor_adaptation="auto", max_cutoff_ratio=1.5)`. It synthesizes one `ModelConfig` and handles different cutoffs and formats. `pipe.make_neighbor_hooks()` returns all the hooks it needs.
* Shorthand `a + b` gives two independent groups whose outputs are summed (`base.py:1072-1120`).

Full AIMNet2 = AIMNet2 + D3(BJ). This follows the pattern of example 08 (`08_aimnet2_ewald_pipeline.py:181-202`):
```python
from nvalchemi.models.dftd3 import DFTD3ModelWrapper           # dftd3.py:435, __init__ :480
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper
d3 = DFTD3ModelWrapper(a1=0.5660, a2=3.1280, s8=0.3908, cutoff=15.0)  # wB97M-D3(BJ) params from Grimme tables (UNVERIFIED; repo docstring only shows PBE a1=0.4289,a2=4.4407,s8=0.7875)
aim_full = PipelineModelWrapper(groups=[PipelineGroup(steps=[aim]), PipelineGroup(steps=[d3])])  # == aim + d3
aim_full.set_config("active_outputs", {"energy", "forces"})
hooks = aim_full.make_neighbor_hooks()     # register at BEFORE_COMPUTE
```
D3 forces are analytic (Warp kernel), and D3 converts Å↔Bohr internally (`dftd3.py:38-43`).
For Transition1x (ωB97x/6-31G(d)) the published AIMNet2 dispersion level is a different functional anyway. Forces from AIMNet2 without D3 may agree better with T1x labels. Test both.

---

## 3. Custom wrapper for Orb (no toolkit wrapper exists)

Contract (`.claude/skills/nvalchemi-model-wrapping/SKILL.md`, `nvalchemi/models/base.py:167` `ModelConfig`, `:337` `BaseModelMixin`):
* Use `class W(nn.Module, BaseModelMixin)`, with `nn.Module` first. Set `self.model_config` in `__init__`, otherwise a `TypeError` is raised at construction (`base.py:383-420`).
* Implement the abstract `embedding_shapes` (property) and `compute_embeddings`. Raising `NotImplementedError` in `compute_embeddings` is fine.
* `forward(data) -> OrderedDict(energy=[B,1], forces=[V,3])`. Optional `adapt_input`/`adapt_output` must call `super()` first.
* `ModelConfig` capability fields: `outputs`, `autograd_outputs`, `autograd_inputs`, `required_inputs`, `optional_inputs`, `supports_pbc`, `needs_pbc`, `neighbor_config`. Runtime fields: `active_outputs`, `gradient_keys`.
* With `neighbor_config=None` no neighbor hook is created (`base.py:1145-1147`). Use this when the model builds its own graph (Orb, the full AIMNet2 calculator).

**Template: `AIMNet2rxnWrapper`** (`examples/advanced/11_batched_neb.py:215-277`). It wraps a complete external calculator. `ModelConfig(outputs={"energy","forces"}, required_inputs={"charge"}, supports_pbc=False, needs_pbc=False, active_outputs={"energy","forces"})` has no `neighbor_config` and no `autograd_outputs`. `forward` packs `{"coord": positions, "numbers": Z, "charge": charge.squeeze(-1), "mol_idx": batch_idx}`, calls `calculator(..., forces=True)`, unsqueezes energy to `[B,1]`, and returns `OrderedDict(energy=..., forces=...)`. The calculator does its own neighbor list and D3/Coulomb.

Orb skeleton in the same pattern (Orb calls are **UNVERIFIED**, check against `orb_models`):
```python
from collections import OrderedDict
import ase, torch
from torch import nn
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig

class OrbWrapper(nn.Module, BaseModelMixin):
    def __init__(self, orbff: nn.Module, device="cuda"):
        super().__init__()
        self.model = orbff.eval()
        self.device_ = torch.device(device)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            supports_pbc=False, needs_pbc=False,
            active_outputs={"energy", "forces"},
        )
    @property
    def embedding_shapes(self): return {}
    def compute_embeddings(self, data, **kw): raise NotImplementedError
    def forward(self, data: AtomicData | Batch, **kw):
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs   # UNVERIFIED
        from orb_models.forcefield.base import batch_graphs                         # UNVERIFIED
        ptr = data.batch_ptr.tolist()
        Z = data.atomic_numbers.detach().cpu().numpy()
        R = data.positions.detach().cpu().numpy()
        graphs = [ase_atoms_to_atom_graphs(ase.Atoms(numbers=Z[a:b], positions=R[a:b]),
                                           system_config=self.model.system_config, device=self.device_)
                  for a, b in zip(ptr[:-1], ptr[1:])]
        g = batch_graphs(graphs)
        with torch.enable_grad():
            out = self.model.predict(g)            # conservative v3 models: "grad_forces"; direct: "forces" (UNVERIFIED)
        e = out["energy"].reshape(-1, 1).to(data.positions)
        f = (out.get("forces", out.get("grad_forces"))).to(data.positions)
        return OrderedDict(energy=e.detach(), forces=f.detach())
# orbff = orb_models.forcefield.pretrained.orb_v3_conservative_inf_omat(device="cuda")  # or an omol variant for molecules (UNVERIFIED)
```
The CPU round-trip is acceptable for ≤23-atom molecules. Keep the atom ordering identical (per-graph slices via `batch_ptr`).

---

## 4. Hooks

Protocol (`nvalchemi/hooks/_protocol.py`; `examples/advanced/02_custom_hook.py:90-97`): any object with `stage`, `frequency`, and `__call__(ctx, stage)`. No base class is needed. `__enter__`/`__exit__` are called around `run()` when present. Multi-stage hooks add `_runs_on_stage(stage)->bool`.

Stages (`nvalchemi/dynamics/base.py:211`, enum `DynamicsStage`):
`ON_ADMISSION(-1)`, `BEFORE_STEP`, `BEFORE_PRE_UPDATE`, `AFTER_PRE_UPDATE`, `BEFORE_COMPUTE`, `AFTER_COMPUTE`, `BEFORE_POST_UPDATE`, `AFTER_POST_UPDATE`, `AFTER_STEP`, `ON_CONVERGE`, `ON_GRADUATE`.
* Frequency gate: a hook fires when `step_count % frequency == 0`, so step 0 always fires (`nvalchemi/hooks/_registry.py:220`). `ON_ADMISSION` and `ON_GRADUATE` ignore frequency.
* `ON_ADMISSION` fires once per (re)admission, i.e. on each `run()` and after every refill. It fires before force priming, at pristine positions (`base.py:1984-2000`, FusedStage `:4848`).
* `ON_GRADUATE`: `ctx.graduated_mask` = graphs active at step start whose status is now `>= exit_status` (`base.py:4770-4800`). The mask can be all False, so always read it. A graph graduated *at ON_ADMISSION* is never "active at step start" and therefore is **not** reported here.

Context (`nvalchemi/hooks/_context.py:35,63`): `DynamicsContext(batch, model, global_rank, workflow, step_count, converged_mask, active_graph_mask, graduated_mask)`. `ctx.workflow` is the engine; use `ctx.workflow.exit_status`. `ctx.batch` holds the latest model outputs (`batch.energy`, `batch.forces`) after `AFTER_COMPUTE`. `ctx.active_graph_mask` is a snapshot from step start; for the current status call `BaseDynamics.active_graph_mask(ctx.batch, exit_status)` (`base.py:1929`).

Registration: `Dyn(..., hooks=[...])` or `dyn.register_hook(hook, stage=None)` (`_registry.py:65`). Hooks at the same stage fire in registration order. On a `FusedStage`, hooks passed to the FusedStage itself are fused-level and wrap the sub-stage hooks.

How systems get "done":
* `ConvergenceHook(criteria=[{"key":"forces","threshold":0.05,"reduce_op":"norm","reduce_dims":-1}], source_status=0, target_status=1, frequency=1)` (`base.py:3592`, stage `AFTER_STEP` `:3679`). The shorthands are `ConvergenceHook.from_fmax(threshold)` (`:3746`) and `from_forces(threshold)` (`:3787`). It migrates status in place with `batch.status.view(-1).copy_(torch.where(...))` (`:3918-3943`). Your hook can do the same.
* Step budget: a sub-stage `n_steps` in a FusedStage is a per-system budget. When it runs out, status jumps to the next code or to exit (`base.py:4705-4747`).
* Built-ins: `NaNDetectorHook` (raises), `MaxForceClampHook(max_force)`, `LoggingHook`, `SnapshotHook(sink, frequency)`, and `ConvergedSnapshotHook(sink)` (`ON_CONVERGE`, `dynamics/hooks/snapshot.py:127`), all from `nvalchemi.dynamics.hooks`. `NeighborListHook` and `HookContext` come from `nvalchemi.hooks`.

### DisagreementHook skeleton (fused-level, every N steps)
```python
import torch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.hooks import DynamicsContext
from nvalchemi.neighbors import compute_neighbors

_NBR = frozenset({"neighbor_list", "neighbor_list_shifts", "neighbor_matrix",
                  "num_neighbors", "neighbor_matrix_shifts"})

class DisagreementHook:
    """Every `frequency` steps: committee force spread per system; graduate the worst ones."""
    stage = DynamicsStage.AFTER_STEP          # or ON_ADMISSION for "screen once at pristine geometry"

    def __init__(self, committee: dict, threshold: float = 0.15, frequency: int = 10, max_flag_frac: float = 1.0):
        self.committee, self.threshold, self.frequency = committee, threshold, frequency
        self.max_flag_frac = max_flag_frac
        self.flagged: dict[int, dict] = {}    # system_id -> {"spread":..., "step":..., "reason":"disagreement"}
        self.screened: set[int] = set()

    def _forces(self, model, sub):
        b = sub.clone()
        b["positions"] = b.positions.detach().clone()
        nc = model.model_config.neighbor_config
        if nc is not None:
            compute_neighbors(b, config=nc)
        with torch.enable_grad():
            out = model(b)
        return out["forces"].detach().to(torch.float64)

    @torch.compiler.disable
    def __call__(self, ctx: DynamicsContext, stage) -> None:
        batch, exit_status = ctx.batch, ctx.workflow.exit_status
        active = BaseDynamics.active_graph_mask(batch, exit_status)      # current status
        if active is None or not bool(active.any()):
            return
        idx = torch.nonzero(active, as_tuple=True)[0]
        _ = batch.batch_ptr                                               # lazy init (snapshot.py:210)
        sub = batch.index_select(idx, drop=_NBR)
        F = torch.stack([self._forces(m, sub) for m in self.committee.values()])   # [M,V,3]
        dev = (F - F.mean(0, keepdim=True)).norm(dim=-1)                 # [M,V] |F_m - F_mean|
        per_atom = dev.pow(2).mean(0).sqrt()                             # [V] RMS over models
        spread = torch.zeros(sub.num_graphs, dtype=per_atom.dtype, device=per_atom.device)
        spread.scatter_reduce_(0, sub.batch_idx.long(), per_atom, reduce="amax", include_self=False)
        flag = spread > self.threshold
        if not bool(flag.any()):
            return
        g = idx[flag]
        status = batch.status.view(-1)
        status[g] = exit_status                                           # evicted at next refill_check
        sids = batch.system_id.view(-1)[g].tolist()
        for sid, s in zip(sids, spread[flag].tolist()):
            self.flagged[sid] = {"spread": s, "step": ctx.step_count, "reason": "disagreement"}
            # also snapshot positions/Z here if the oracle needs the exact flagged geometry:
        pos = batch.positions.detach()
        for sid, gi in zip(sids, g.tolist()):
            a, b = batch.batch_ptr[gi].item(), batch.batch_ptr[gi + 1].item()
            self.flagged[sid].update(Z=batch.atomic_numbers[a:b].cpu(), R=pos[a:b].cpu())
```
* The driver model (the one the dynamics uses) can be one of the committee members. Its forces are already in `batch.forces` at `AFTER_STEP`, so you can reuse them and skip one forward pass.
* Register on the **FusedStage** so `ctx.workflow.exit_status` is the fused exit code. Pass it via `FusedStage(..., hooks=[hook])` or `fused.register_hook(hook)`.
* Screen-once variant: use `stage = DynamicsStage.ON_ADMISSION`, skip graphs whose `system_id` is in `self.screened`, then add them. Positions at that point are the dataset geometry, which matters when the oracle is a Transition1x lookup.
* Alternative to calling models in a hook: a `CommitteeWrapper(BaseModelMixin)` that returns mean `energy`/`forces` plus an extra output `force_spread [B,1]`. Unregistered outputs are published to the batch (`base.py:2958-3000`), and a 3-line hook then thresholds `batch.force_spread`.

---

## 5. In-flight batching

Pieces (`examples/intermediate/04_inflight_batching.py`):
```python
from nvalchemi.dynamics import FIRE2, SizeAwareSampler, FusedStage, ConvergenceHook   # dynamics/__init__.py:22-41
```
Dataset protocol (`sampler.py:52-55`, example `:148-213`): `__len__`, `get_metadata(idx) -> (num_atoms, num_edges)` (it must be cheap because it is called for every sample at construction), and `__getitem__(idx) -> (AtomicData, dict)`. The batch device follows the AtomicData device (`sampler.py:362`), so return CUDA tensors.

```python
class T1xDataset:
    def __init__(self, items, device="cuda"):       # items: list of (Z, R, E, F, key)
        self.items, self.device, self.load_order = items, device, []
    def __len__(self): return len(self.items)
    def get_metadata(self, i): return len(self.items[i][0]), 0      # edges 0: no max_edges constraint
    def __getitem__(self, i):
        Z, R, E, F, _ = self.items[i]
        self.load_order.append(i)                    # system_id k  <->  load_order[k]  (stamp happens right after getitem)
        return to_atomic_data(Z, R, E, F, device=self.device), {}

ds = T1xDataset(items)
sampler = SizeAwareSampler(ds, max_atoms=4096, max_edges=None, max_batch_size=256,
                           bin_width=1, shuffle=False, max_gpu_memory_fraction=0.8)   # sampler.py:94
```
* `system_id` is a monotonic **load counter**, not a dataset index (`sampler.py:334-339`, `:611-616`). Map it back with `load_order` as above, or add your own system property in `__getitem__` (`data.add_system_property("t1x_idx", torch.tensor([[i]]))`).
* `max_atoms`: the effective cap is `min(max_atoms, GPU estimate)` when CUDA is available (`sampler.py:279-286`). `RuntimeError` is raised if any sample exceeds the caps.
* Alternatively, `nvalchemi.data.datapipes.InMemoryDataset(in_memory_batch=Batch, device=...)` implements `get_metadata` and `__getitem__` too (`in_memory_dataset.py:574,590`).

Engine (Mode 2):
```python
driver = mace                                         # one BaseModelMixin
relax = FIRE2(model=driver, dt=0.05, n_steps=200,     # n_steps = per-system budget inside FusedStage
              convergence_hook=ConvergenceHook.from_fmax(0.05))
for h in driver.make_neighbor_hooks():
    relax.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
dis = DisagreementHook({"mace": mace, "aimnet2": aim, "orb": orb}, threshold=0.15, frequency=10)
graduates = GraduateCollector()                       # see below
fused = FusedStage(sub_stages=[(0, relax)], sampler=sampler, sinks=[], refill_frequency=1,
                   hooks=[dis, graduates])            # exit_status auto = 1
with fused:
    result = fused.run(batch=None, n_steps=10_000)    # None => sampler exhausted, all graduated
```
* What refill does (`base.py:3305-3445`): graduated graphs (status `>= exit_status`) are written to `sinks` (if any), the survivors are `index_select`ed, and replacements are requested up to the remaining atom, edge and count budget. Then `status=0` and `reprime_pending=True` are set for newcomers, and the result is a **new Batch object**. Sinks must have enough capacity, otherwise "All sinks are full" `RuntimeError` (`base.py:1215-1238`).
* `FusedStage(..., init_fn=callable)` runs once on the sampler-built initial batch (`base.py:4090-4095`).
* Collecting results with a stream-out hook (better than sinks, because sinks lose custom keys):
```python
class GraduateCollector:
    stage, frequency = DynamicsStage.ON_GRADUATE, 1
    def __init__(self): self.records = []
    @torch.compiler.disable
    def __call__(self, ctx, stage):
        m = ctx.graduated_mask
        if m is None or not bool(m.any()): return
        b = ctx.batch
        fmax = torch.zeros(b.num_graphs, device=b.positions.device).scatter_reduce_(
            0, b.batch_idx.long(), b.forces.norm(dim=-1), reduce="amax", include_self=False)
        for gi in torch.nonzero(m, as_tuple=True)[0].tolist():
            a, e = b.batch_ptr[gi].item(), b.batch_ptr[gi + 1].item()
            self.records.append(dict(system_id=int(b.system_id.view(-1)[gi]), fmax=float(fmax[gi]),
                energy=float(b.energy[gi]), R=b.positions[a:e].detach().cpu(), Z=b.atomic_numbers[a:e].cpu()))
```
* Telling causes apart: if `system_id in dis.flagged` it was ours; else if `fmax <= 0.05` it converged; else the step budget ran out. Hook-graduated systems ARE reported at `ON_GRADUATE` when flagged at `AFTER_STEP`, but not when flagged at `ON_ADMISSION`, so use `dis.flagged` for those. You can also stream to an oracle queue directly from inside `DisagreementHook`.
* Built-in sinks: `HostMemory(capacity)`, `GPUBuffer(capacity, max_atoms, max_edges, device)`, `ZarrData(store, capacity)` (`nvalchemi/dynamics/sinks.py:514,225,650`). `sink.drain()` returns a Batch.

---

## 6. Relaxation (no in-flight)

```python
from nvalchemi.dynamics import FIRE2, LBFGS, ConvergenceHook
from nvalchemi.dynamics.base import DynamicsStage
opt = FIRE2(model=mace, dt=0.05, n_steps=300,                         # fire2.py:155; example basic/02:170
            convergence_hook=ConvergenceHook.from_fmax(0.05), hooks=[])
# opt = LBFGS(model=mace, history_size=6, maxstep=0.2, n_steps=2000, convergence_hook=...)  # lbfgs.py:306; basic/06:100
for h in mace.make_neighbor_hooks():
    opt.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)           # REQUIRED for MACE/AIMNet2
batch = Batch.from_data_list(datas, device="cuda")                     # with forces/energy/velocities
with opt:
    batch = opt.run(batch)                                             # base.py:3243; early exit when all converge
print(batch.status.view(-1))   # 1 = converged (exit_status default 1), 0 = not
```
`run` returns the batch. A standalone FIRE2 does not refill; converged graphs stay in the batch as no-ops (`base.py:3075-3080`).

---

## 7. Batched NEB (`examples/advanced/11_batched_neb.py`, `nvalchemi/dynamics/mep/`)

```python
from nvalchemi.dynamics.mep import NEB, ClimbingImageConfig, IDPPModel, NEBMethod, interpolate_paths, prepare_idpp_targets
initial = Batch.from_data_list([AtomicData(positions=R0, atomic_numbers=Z, charge=torch.tensor([[q]])) ...], device=dev)  # :120-138
final   = Batch.from_data_list([... product ...], device=dev)
band = interpolate_paths(initial, final, 10)            # interpolate.py:125 (graph i of both batches = path i); sets group_layout
idpp_band = prepare_idpp_targets(band.clone())          # idpp.py:68 (full pair list + target distances)
init = lambda b: (setattr(b, "energy", torch.zeros(b.num_graphs, 1, device=b.device)),
                  setattr(b, "forces", torch.zeros_like(b.positions)),
                  setattr(b, "velocities", torch.zeros_like(b.positions)))   # :166-176
init(idpp_band)
idpp_band = NEB(model=IDPPModel(), fmax=0.1, n_steps=100).run(idpp_band)     # :188-193; status==1 => converged
neb_band = band.clone(); neb_band.positions.copy_(idpp_band.positions); init(neb_band)
neb = NEB(model=model, spring=0.1, method="improved_tangent", fmax=0.05, n_steps=500,
          climbing=ClimbingImageConfig(regular_fmax=0.5), diagnostics_log_path=Path("neb.csv"))  # :355-363
neb_band = neb.run(neb_band)     # status 0 regular, 1 climbing, 2 done
E = neb_band.energy.reshape(n_paths, n_images); rel = E - E[:, :1]          # :402-407
```
* `NEB` is a Pydantic `DynamicsStrategy` (`mep/neb.py:157`). Its fields: `spring`, `method`, `climbing`, `optimizer=FIRE2` (or `LBFGS`), `optimizer_kwargs`, `fmax`, `endpoint_mode="fixed"`, `fixed_atom_indices`, `diagnostics_log_path`, `diagnostics_frequency`, `compile`, `neighbor_hooks` (default `model.make_neighbor_hooks()`), and `convergence_hook`. It also has `model`, `n_steps`, and **`extra_hooks`** (`dynamics/strategy.py:123-157`), so DisagreementHook can be attached to NEB too.
* Model used there: `AIMNet2rxnWrapper` (§3), the full `aimnet2-rxn` calculator including Coulomb and D3.
* Data files (`examples/advanced/batched_neb_endpoints/`), 8 Transition1x v4 reactions:
  * `endpoints.json`: `{"source": {...}, "paths": [{"reaction_id": "C2H2N2O/rxn2093", "charge": 0, "atomic_numbers": [7 ints], "reactant_positions": [[x,y,z]*n], "product_positions": [[...]]}, ...]}`.
  * `dft_reference.json`: `{"source": {...}, "paths": [{"reaction_id", "atomic_numbers", "positions": [10 images][n][3], "relative_energies_ev": [10], "transition_state_index": int}]}`.
* DFT comparison (`:434-469`): both are relative to image 0 (the reactant). `dft_barrier = rel_dft[ts_index]` and `model_barrier = max(rel_model)`. Profiles are plotted against normalized cumulative Cartesian path length. Reference level: ωB97x/6-31G(d). Runtime: ~40–70 s on one L4 (`:66-67`).

---

## 8. LoRA fine-tuning + acceptance report

Imports (`examples/intermediate/08_lora_finetuning.py:70-99`):
```python
from nvalchemi.data.datapipes import DataLoader, InMemoryDataset
from nvalchemi.hooks import HookContext, NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.training import (FineTuningStrategy, OptimizerConfig, ValidationConfig, TrainingStage,
    default_training_fn, EnergyHuberLoss, ForceHuberLoss, EnergyMSELoss, ForceMSELoss)
from nvalchemi.training.peft import load_peft_checkpoint_into_model          # peft/loading.py:62
from nvalchemi.training.peft.lora import LoRAConfig, available_lora_wrappers, is_lora_layer, merge_lora_into_model
from nvalchemi.training.reference_energies import fit_atomic_reference_energies   # reference_energies.py:68
```
**Dataset format: in-memory is fine, no Zarr needed.** Build `AtomicData(energy=[[E]], forces=F, ...)` per oracle-labelled structure, then `Batch.from_data_list`, then `InMemoryDataset`, then `DataLoader`. Losses read targets from `batch.energy` / `batch.forces`.
```python
train_b = Batch.from_data_list(train_datas); val_b = Batch.from_data_list(val_datas)
def loader(b, bs, shuffle):
    return DataLoader(InMemoryDataset(in_memory_batch=b, device=device), batch_size=bs, shuffle=shuffle,
                      drop_last=shuffle, prefetch_factor=0, num_streams=1, use_streams=False, pin_memory=True)  # :241-261
```
E0 re-referencing (essential: T1x is ωB97x total energies while MACE-MPA-0 is PBE materials). See `:292-418`:
```python
base = MACEWrapper.from_checkpoint("medium-mpa-0", device=device, dtype=torch.float32)
base.model_config.active_outputs = {"energy"}; base.eval()
nl = NeighborListHook(base.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD)
def baseline(b):
    nl(HookContext(batch=b, model=base), TrainingStage.BEFORE_FORWARD); return base(b)["energy"]
fit = fit_atomic_reference_energies(loader(train_b, 64, False), baseline_energy_fn=baseline, device=device)
ckpt_e0 = dict(zip(base.model.atomic_numbers.tolist(), base.model.atomic_energies_fn.atomic_energies.reshape(-1).tolist()))
e0 = {z: ckpt_e0[z] + d for z, d in fit.reference_energies.items()}     # residual added to checkpoint E0
```
Strategy (`:426-567`; `FineTuningStrategy` at `training/finetune.py:117`, `LoRAConfig` at `training/peft/lora.py:75`, fields `rank=8, alpha=1.0, lora_target_patterns (required), lora_dropout=0.0, wrap_mlp=False`):
```python
model = MACEWrapper.from_checkpoint("medium-mpa-0", device=device, dtype=torch.float32, enable_cueq=True,
                                    compile_model=False, atomic_energies=e0, train=True)
model.model_config.active_outputs = {"energy", "forces"}; model.train()
loss_fn = 1.0 * EnergyHuberLoss(per_atom=True, delta=0.01, ignore_nonfinite=True) \
        + 10.0 * ForceHuberLoss(normalize_by_atom_count=False, delta=0.01, ignore_nonfinite=True)
loss_fn.normalize_weights = False
strategy = FineTuningStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW, optimizer_kwargs={"lr": 5e-4, "weight_decay": 1e-4}),
    num_epochs=20, num_steps=None, training_fn=default_training_fn, loss_fn=loss_fn, devices=[device],
    hooks=[NeighborListHook(model.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD)],
    validation_config=ValidationConfig(validation_data=loader(val_b, 64, False), validation_fn=default_training_fn,
                                       loss_fn=loss_fn, every_n_steps=50, grad_mode="auto", use_ema="auto", name="validation"),
    peft_config=LoRAConfig(rank=16, alpha=2.0, lora_dropout=0.0, lora_target_patterns=(
        "main.model.node_embedding.linear", "main.model.interactions.0.linear",
        "main.model.interactions.*.linear_up", "main.model.interactions.*.conv_tp_weights.layer[123]",
        "main.model.products.*.linear", "main.model.readouts.*.linear",
        "main.model.readouts.*.linear_1", "main.model.readouts.*.linear_2")),          # :130-139
    trainable_patterns=("main.model.interactions.1.linear.weight", "main.model.interactions.*.density_fn.layer0.weight"),
)
strategy.run(loader(train_b, 64, True))
idx = strategy.save_checkpoint("runs/peft", save_trainable_state_only=True)      # strategy.py:1585
tuned = strategy.models["main"]          # live adapted model; or merge_lora_into_model(tuned) (lora.py:263) to fold adapters in
# reload later:
fresh = MACEWrapper.from_checkpoint("medium-mpa-0", device=device, dtype=torch.float32, enable_cueq=True, atomic_energies=e0)
fresh = load_peft_checkpoint_into_model(fresh, "runs/peft")   # merge=True default; model_name="main"
```
* `default_training_fn(model, batch)` returns `{"predicted_energy", "predicted_forces", ...}` (`strategy.py:257-276`).
* The model is stored under key `"main"`, so patterns are prefixed with `main.model.`. The target patterns were written for a cuEq-converted MACE (`ENABLE_CUEQ=True`). If you run with `enable_cueq=False`, print `[n for n,_ in model.named_modules()]` and `available_lora_wrappers()` and re-check the patterns (**UNVERIFIED** for e3nn layout).
* Losses (`nvalchemi.training`): `EnergyMSELoss`, `ForceMSELoss(normalize_by_atom_count=...)`, `EnergyHuberLoss`, `ForceHuberLoss`, `ForceL2NormLoss`, `StressHuberLoss`. They compose with `w*A + w*B`. Set `loss_fn.dtype_policy="prediction_to_target"` if dtypes differ.
* Time and memory: the only number in the repo is from the example. 100 epochs on 540 LPSC structures on an L4 reached an E MAE of 0.36 meV/atom and an F MAE of 0.0136 eV/Å (`:704-712`). No memory figures are given. For a few hundred ≤23-atom molecules, expect seconds per epoch.

Acceptance report (`nvalchemi/training/distillation/evaluation/`, `report.py:101,232,693,1056`, `accuracy.py:690`):
```python
from nvalchemi.training.distillation.evaluation import (AcceptanceThresholds, StudentEvaluation,
    build_acceptance_report, evaluate_accuracy)
# holdout must be RE-ITERABLE and each batch must already CARRY a neighbor list (accuracy.py:815-819)
hold = []
for b in loader(test_b, 64, False):
    compute_neighbors(b, config=tuned.model_config.neighbor_config); hold.append(b)
acc_before = evaluate_accuracy(base_with_e0, hold, targets="reference", quantities=("energy", "forces"))
acc_after  = evaluate_accuracy(tuned,        hold, targets="reference", quantities=("energy", "forces"))
report = build_acceptance_report(
    [StudentEvaluation(name="mace_lora", accuracy=acc_after, baseline_accuracy=acc_before, weights="raw")],
    AcceptanceThresholds(max_forces_mae=0.10, max_energy_per_atom_mae=0.02, max_from_scratch_ratio=1.0))
report.accepted                     # bool verdict (all students, all bars)
report.verdicts[0].checks           # AcceptanceCheck(name, value, limit, comparison, passed, detail)
report.to_dict(); report.scalars()  # JSON / flat metrics;  rich.Console().print(report) for a table
```
* `AccuracyMetrics` fields include `energy_mae/rmse`, `energy_per_atom_mae/rmse`, `forces_mae/rmse`, `force_cosine_mean/aggregate`, `num_graphs`, and `num_atoms` (`accuracy.py:187-226`), all in batch units.
* `max_from_scratch_ratio` takes the worst ratio student/baseline over the energy-per-atom, force, and stress MAEs (`report.py:831-910`). With the pre-fine-tune model as the "baseline", `1.0` means "must not regress". Both evaluations must use the same holdout (same graph and atom counts), otherwise the check fails.
* A bar you set without a matching measurement **fails** the student (`report.py:232-245`). Leave the stability and RDF bars unset.
* Other thresholds: `min_force_cosine`, `max_stress_mae`, drift and momentum bars, `no_divergence`, `max_extensivity_error_per_atom`, `max_rdf_jensen_shannon`, and `extra` for custom bars.

---

## 9. Gotchas

* **Neighbor lists are never automatic.** A dynamics run without `make_neighbor_hooks()` registered gives MACE no `neighbor_list` (an `AttributeError`/None error in `adapt_input`). Training needs `NeighborListHook(..., stage=TrainingStage.BEFORE_FORWARD)`. `evaluate_accuracy` needs batches that already carry neighbors.
* **Neighbor format differs per model:** MACE uses COO and AIMNet2 uses MATRIX. A `PipelineModelWrapper` plans both. For a hand-rolled committee, clone the batch per model.
* **dtype:** AIMNet2 is fp32 only and mutates the batch positions to fp32. MACE casts the input to the model dtype. `Batch.put` (Warp buffers) handles float32 only (dynamics-api skill, "Buffer lifecycle"). Keep everything float32.
* **Autograd forces inside hooks:** wrap model calls in `torch.enable_grad()`. Hooks run inside `requires_grad_ctx` during the FusedStage step, but `ON_ADMISSION` runs outside it. Detach outputs.
* **Charge:** set `charge=[[0.]]` on every AtomicData. AIMNet2 declares it required but silently defaults to 0. `from_atoms` wants an int `atoms.info["charge"]`.
* **Non-periodic:** leave `cell`/`pbc` as None. MACE builds an identity cell and zero shifts. MACE `supports_pbc=True, needs_pbc=False`, AIMNet2 likewise. No vacuum box is needed.
* **Energy referencing:** MACE-MPA-0 (PBE/PBE+U, materials) versus T1x ωB97x/6-31G(d) totals versus AIMNet2 (ωB97M). Absolute energies are not comparable. Use **forces** for disagreement, or per-system relative energies. Refit E0s before fine-tuning (§8).
* **`system_id` is a load counter, not a dataset index**, and it is `-1` for batches built outside the sampler (`base.py:1792-1794`).
* **Custom batch keys vs sinks:** they are lost through `HostMemory` (to_data_list round trip). Read them in hooks.
* **Mode 2 with a final stage that has no convergence and no `n_steps`** loops until `n_steps` on `run()` (`base.py:5017-5021`). Always pass `n_steps`.
* **Sinks capacity:** if `sinks=[...]`, total capacity must be ≥ the number of graduates, otherwise `RuntimeError`. With `sinks=[]` graduates are simply dropped after `ON_GRADUATE`.
* **Status written at `ON_ADMISSION`** evicts the system at the next refill but skips `ON_GRADUATE`.
* **`refill_check` returns a new Batch.** Never hold references to an old batch across steps in hooks. Index by `system_id`.
* **CUDA:** `cu13` is the Makefile default. Use `uv sync --extra cu12 --extra mace` (or `--extra cu13`) plus `--extra aimnet --extra ase`. Never use `--all-extras`: the CUDA variants are mutually exclusive, and `uma*` conflicts with `mace` (AGENTS.md "CUDA And Environment Setup"). The `mace` extra pins `mace-torch==0.3.15` (`pyproject.toml`). cuEq (`enable_cueq=True`) needs `cuequivariance-ops-torch-cu12/13`, which the `cu12`/`cu13` extras pull in, and a CUDA device. On CPU or Windows without it, use `enable_cueq=False`. The CUDA extras are skipped on macOS only.
* **Python** `>=3.11,<3.15` (`pyproject.toml:12`).
* **MACE `train=False` (default) freezes the weights.** That is fine for inference forces, but pass `train=True` for fine-tuning. Compiled fine-tuning is not covered.
* **`ConvergenceHook` defaults:** `ConvergenceHook()` uses forces norm ≤ 0.05 (`base.py:3685-3689`). `from_fmax` is the same as `from_forces`.
* **Orb has no wrapper and no `to_atoms` helper** in the toolkit. Build ASE `Atoms` per graph from `batch_ptr` slices yourself.
