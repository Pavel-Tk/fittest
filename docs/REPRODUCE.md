# Reproducing TrustSwarm

Everything we report was produced on a single **NVIDIA L4 (24 GB), CUDA 13 driver** (Google Colab),
with the commands below. Runtimes are wall-clock numbers from our job logs. The CPU parts
(agents, ablation, figures, trust service) also run on a laptop from the cached predictions.

## 0. Environment

```bash
git clone <this repo> trustswarm && cd trustswarm
bash scripts/setup_gpu.sh ..          # toolkit 0.3.0-rc + core env, Orb env, Transition1x (6.6 GB, md5-checked)
```

`setup_gpu.sh` does exactly what we did:

* **Core env** (on Colab this was the system `python3`, 3.13):
  `uv pip install -e "nvalchemi-toolkit[cu13,mace,aimnet,ase]"` at tag `0.3.0-rc` with
  `--extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match`,
  then `h5py scikit-learn matplotlib fastapi uvicorn httpx pytest`. We got torch 2.11+cu130,
  mace-torch 0.3.15, aimnet 0.2.5 and cuEquivariance. `CORE_VENV=1` installs into a fresh venv instead.
* **Orb env**: a Python 3.12 venv with `"dm-tree>=0.1.9" orb-models ase h5py numpy torch` from the
  cu130 index. orb-models pins different dependencies from the toolkit, so it runs in its own
  process and environment. That is the reason for the model-server design.
* **Transition1x**: `curl -L -A 'Mozilla/5.0' https://ndownloader.figshare.com/files/36035789`.
  figshare rejects requests that have no browser user agent. md5 `faab0413cf6b042b6d50e6272dea9a8c`.

Then export the locations. Every script reads them as CLI defaults (`trustswarm/paths.py`):

| variable | meaning | default |
|---|---|---|
| `TRUSTSWARM_T1X` | Transition1x HDF5 | `data/Transition1x.h5` |
| `TRUSTSWARM_CORE_PYTHON` | interpreter with torch + nvalchemi + mace + aimnet | the running interpreter |
| `TRUSTSWARM_ORB_PYTHON` | interpreter of the Orb env | core python |
| `TRUSTSWARM_NEB_DIR` | toolkit `examples/advanced/batched_neb_endpoints` | found next to an editable `nvalchemi` install |
| `TRUSTSWARM_LOG_DIR` | model-server logs | `logs` |

Our Colab values are in `colab/env.sh`. Each GPU script starts one model server per model
(`trustswarm/model_server.py`) on `--base-port + k`. Pick distinct base ports if several runs
share a machine. `spawn` refuses a port that is already in use.

**Model aliases.** `aimnet2` is the `aimnet` registry alias for ensemble member 0
(`aimnet2-wb97m-d3_0`). The servers `aimnet2` and `aimnet2_m0` give **identical** predictions:
force RMSE vs DFT on all 5632 frames is 0.316169 eV/Å for both. Arm A (single model) and
member 0 of arm B are the same network.

## 1. Pipeline (in order)

All commands run from the repo root with the core interpreter (`python` below).

| step | command | outputs | L4 runtime |
|---|---|---|---|
| 1. frames + per-model predictions | `python scripts/predict_pool.py --build-frames --models aimnet2,aimnet2_m0,aimnet2_m1,aimnet2_m2,aimnet2_m3,orb,mace_mpa0 --base-port 6000` | `data/frames.pkl` (5632 frames), `data/preds/{model}.npz`, `results/predict_status.json` | ~2.5 min in total. Per model incl. server start: aimnet2 14–16 s (~80k atoms/s), orb 25 s (3.5k atoms/s), mace_mpa0 54 s (2.7k atoms/s) |
| 2. is disagreement a signal? | `python scripts/analyze_signal.py --frames data/frames.pkl --preds data/preds` | `results/signal.{json,png}` | seconds (CPU) |
| 3. in-flight ALCHEMI screening | `python scripts/screen_alchemi.py --split pool --max-frames 600 --every 10 --n-steps 50 --score-labels --base-port 6100` | `results/screen_alchemi.json`, `data/oracle_queue_alchemi.pkl` | 131 s run + ~60 s for 3 servers to start (600 frames, 1500 atom-steps/s) |
| 4. batched NEB with the hook | `python scripts/run_neb.py --driver aimnet2 --base-port 6103` | `results/neb.json`, `results/neb_sigma.png` | NEB 21 s + ~60 s server start |
| 5. ablation, arms A–D | `python scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/ --budget-per-round 40 --rounds 5 --seed 0` | `results/ablation.{json,png}`, `blackboard_armD.jsonl`, `trust_snapshot.json`, `oracle_labels_armD.json` (the Curator's repair request) | ~12 s (CPU) |
| 6. LoRA fine-tune of the worst model on the oracle labels | `python scripts/finetune.py --frames data/frames.pkl --labelled-uids results/oracle_labels_armD.json --model mace_mpa0 --out data/preds/mace_lora_curated.npz --ckpt data/ckpt/mace_lora_curated --epochs 40 --seed 0 --predict-uids all` | `data/preds/mace_lora_curated.npz`, `data/ckpt/mace_lora_curated/` (incl. `acceptance_report.json`) | ~2 min (75 s training) |
| 7. ablation again (arm D replays the fine-tune) | same command as step 5 | updated `results/ablation.*` | ~12 s |
| optional: seed-ensemble B′ | step 6 with `--seeds 0,1,2,3` | `data/preds/mace_seed{0..3}.npz` | ~6 min |
| figures | `python scripts/make_figures.py` | `results/fig/*` | seconds (CPU) |

Steps 2, 5, 7 and the figures need only numpy/scipy/sklearn/matplotlib. Copy `data/` and
`results/neb.json` to a laptop and run them there.

Checks we ran: `python -m pytest -q` (CPU, 76 tests); on the GPU, `python -m pytest tests/test_hook.py`
(7 tests, including a real `FusedStage` + `SizeAwareSampler` + FIRE2 run); and
`python scripts/check_mace_server.py --base-port 6100` (MACE-MPA-0 server: force RMSE 0.58 eV/Å on 25 T1x test frames).
`screen_alchemi.py` also checks that the in-flight hook's sigma at admission matches the offline
`data/preds` + `signals.force_spread` value. On 600 frames the largest difference was 2.8e-6 eV/Å.

## 2. Use the DisagreementHook in your own Toolkit run

`trustswarm.hook.DisagreementHook` is a plain Toolkit hook (`stage`, `frequency`, `__call__(ctx, stage)`,
`_runs_on_stage`). Committee members can be any mix of:

* `ModelClient`s from `trustswarm.model_server.spawn`: one process per model, can use another Python env;
* objects with `.predict(Z, R, batch[, charge])` that return `{"energy", "forces"}` or `(E, F)`;
* plain callables with the same contract.

Inputs and outputs are flat numpy arrays: `Z (N,)`, `R (N,3)` Å, `batch (N,)`, then `E (S,)` eV and `F (N,3)` eV/Å.

```python
import torch
from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage, SizeAwareSampler
from nvalchemi.dynamics.base import DynamicsStage

from trustswarm.hook import DisagreementHook
from trustswarm.model_server import spawn
from trustswarm.models.mace_batched import BatchedMACE
from trustswarm.models.toolkit_io import atomic_data
from trustswarm import paths

class MyDataset:                                    # Toolkit sampler protocol
    def __init__(self, mols): self.mols, self.load_order = mols, []     # mols: list of (Z, R) numpy
    def __len__(self): return len(self.mols)
    def get_metadata(self, i): return len(self.mols[i][0]), 0
    def __getitem__(self, i):
        self.load_order.append(i)                   # sampler stamps system_id = position in load_order
        Z, R = self.mols[i]
        return atomic_data(Z, R, device="cuda", with_dynamics_fields=True), {}

ds = MyDataset(mols)
committee = {m: spawn(m, 6100 + k, python=paths.model_python(m)).connect()
             for k, m in enumerate(["mace_mpa0", "aimnet2", "orb"])}
flagged = []
hook = DisagreementHook(committee, tau=0.56,        # eV/Å; a float or a zero-arg callable read at every check
                        every=10, stage="both",     # pristine geometry at admission + every 10 steps
                        graduate=True,              # flagged -> status = exit_status -> sampler refills the slot
                        id_map=lambda sid: ds.load_order[sid],   # ledger key = your dataset index
                        on_flag=lambda rec: flagged.append(rec["key"]))

driver = BatchedMACE().model                        # any Toolkit BaseModelMixin drives the dynamics
relax = FIRE2(model=driver, dt=0.05, n_steps=50, convergence_hook=ConvergenceHook.from_fmax(0.05))
for h in driver.make_neighbor_hooks():
    relax.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
sampler = SizeAwareSampler(ds, max_atoms=1024, max_edges=None, max_batch_size=48)
fused = FusedStage(sub_stages=[(0, relax)], sampler=sampler, sinks=[], refill_frequency=1, hooks=[hook])
with fused:
    fused.run(batch=None, n_steps=20000)

rec = hook.records[flagged[0]]     # sigma_admission, checks[{step, sigma, sigma_max, tau, energy{model}}],
                                   # flagged, flag_step, sigma_at_flag, at_flag{Z, R, forces{model}}, n_steps, reason
print(hook.summary())              # n_flagged, round trips, gather time, dead models, failures
for c in committee.values(): c.close()
hook.close()
```

* NEB, or any run where systems must not leave: `NEB(..., extra_hooks=[DisagreementHook(..., graduate=False)])`.
  Without a sampler the ledger key is the graph index. See `scripts/run_neb.py`.
* No dynamics at all: `trustswarm.hook.DisagreementMonitor(committee, tau).check(keys, Z, R, batch, step=0, phase="x")`.
* Sigma is defined as `sigma_F = mean_i sqrt(mean_m ||F_m,i - mean_m F_m,i||^2)`, identical to
  `trustswarm.signals.force_spread` (tested in `tests/test_hook.py`).
* Robustness: a member that raises or times out is logged in `hook.failures` and retired after
  `max_failures` consecutive failures. Checks continue while at least 2 members answer.

## 3. Add your own model agent

A model agent is a `Backend` in `trustswarm/model_server.py` that runs in its own process:

```python
class MyBackend(Backend):
    level_of_theory = "wB97X-D/def2-TZVP"           # reported by the "info" command

    def __init__(self, device: str):
        self.name = "mymodel"
        self.model = load_my_model(device)

    def predict(self, Z, R, batch, charge):          # Z (N,) int, R (N,3) Å, batch (N,), charge (S,)
        E, F = self.model(Z, R, batch, charge)
        return np.asarray(E, np.float64).reshape(-1), np.asarray(F, np.float64).reshape(-1, 3)  # eV, eV/Å
```

Register it in `make_backend` (`if model == "mymodel": return MyBackend(device)`). Then
`spawn("mymodel", port, python=<its env's interpreter>)` serves it, and it can be passed to
`DisagreementHook`, `predict_pool.py --models ...,mymodel` and the ablation like any other member.
If the model needs its own environment, point `python=` at that interpreter, as we do for Orb
via `TRUSTSWARM_ORB_PYTHON`. The repo must be importable there (`spawn` sets `PYTHONPATH`), and it
needs `numpy`.
