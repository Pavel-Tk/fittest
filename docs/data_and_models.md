# Fittest: data and models reference

Verified 2026-10-30 against source (GitLab/GitHub/HF/figshare API). Items marked **[unverified]** could not be confirmed from primary source in the time available.

Package versions at time of writing: `orb-models 0.7.0` (2026-05-26), `aimnet 0.2.5`, `mace-torch 0.3.17`.

---

## A. Transition1x HDF5

### A.1 Download

- The canonical repo is **https://gitlab.com/matschreiner/Transition1x**. The GitHub path `Schreiner/Transition1x` returns 404. Copies of its dataloader live in ORNL/HydraGNN and valence-labs/OpenQDC.
- URL (from `download_t1x.py`, and also in OpenQDC `__links__`): **https://figshare.com/ndownloader/files/36035789**
- figshare article `19614657`, file `Transition1x.h5`: **6,623,929,000 bytes (6.62 GB)**, md5 `faab0413cf6b042b6d50e6272dea9a8c`.
- figshare can return `HTTP 202` with an empty body to `curl -I` or bot user agents. Use the official `urllib.urlretrieve` call, or wget with a user agent:

```bash
# Colab (local disk is fine; do NOT download to laptop)
wget -q --user-agent="Mozilla/5.0" -O /content/transition1x.h5 https://figshare.com/ndownloader/files/36035789
echo "faab0413cf6b042b6d50e6272dea9a8c  /content/transition1x.h5" | md5sum -c
```

### A.2 Layout (confirmed from `scripts/combine_dbs.py` + `transition1x/dataloader.py`)

```
/data/<formula>/<rxn>/                       # e.g. /data/C2H2N2O/rxn0012   (rxn ids: rxn0000..rxn10072, zero-padded 4+)
    atomic_numbers            (n_atoms,)           int
    positions                 (n_frames, n_atoms, 3)  Å
    wB97x_6-31G(d).energy     (n_frames,)          eV, TOTAL electronic energy
    wB97x_6-31G(d).forces     (n_frames, n_atoms, 3)  eV/Å
    reactant/          {atomic_numbers, positions(1,n,3), wB97x_6-31G(d).energy(1,), wB97x_6-31G(d).forces(1,n,3), hash}
    transition_state/  {same, 1 frame}
    product/           {same, 1 frame}
/train/<formula>  -> h5py.SoftLink("/data/<formula>")
/val/<formula>    -> SoftLink
/test/<formula>   -> SoftLink
```

- **There is no `atomization_energy` dataset.** The dataloader computes it as `E - sum(REF[Z])` with `REF = {1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045, 9: -2712.8213146878606}` (eV).
- **Path frames:** the reaction-level arrays hold the saved NEB/CI-NEB images, in chunks of 10 images per saved iteration. The first chunk keeps both endpoints, so frame 0 is the reactant and frame 9 is the product. Later chunks drop the endpoints (`path[1:-1]`). Iterations are thinned with a cumulative-fmax rule (`cum_fmax < 0.1` → skip next), and the last iteration is always kept.
- `reactant` = `path[0]`, `product` = `path[-1]` of the first iteration. `transition_state` = the argmax-energy image of the **final** iteration (CI-NEB). Each of the three also appears as a duplicate inside the path arrays.
- Scale (paper): 10,073 converged reactions out of 11,961 Grambow reactions, about 9.6M DFT single points. Elements are H, C, N, O, with at most 7 heavy atoms. All species are neutral and closed-shell.
- **Official split = by chemical formula (composition), not by reaction.** Val and test are each about 5% of the data. No test formula appears in train. Hard-coded lists in `combine_dbs.py`:
  - `test_mols = ["C3H5NO2","C2H5NO2","C2H3N3O2","C2HNO3","C3H8N2O","C3HNO2","C5H5NO","C2HNO"]`
  - `val_mols = ["C3N2O","C2H2N2O","C4H10N2O","C3H6N2O","C4H9NO2","CHN3O","C2H6N2O","CN2O3"]`
  - train: 155 other formulas. Sub-groups are SoftLinks, so iterating `f["test"]` works directly.
  - Do NOT confuse this with the OA-ReactDiff/React-OT random split (9,000 train / 1,073 test reactions) that Zhao et al. 2025 use.

### A.3 Reader (h5py, per-frame, N frames per reaction)

```python
import h5py, numpy as np

KE = "wB97x_6-31G(d).energy"; KF = "wB97x_6-31G(d).forces"

def iter_t1x(path, split="test", n_path=8, seed=0, include_endpoints=True, formulas=None):
    """Yield dict(reaction_id, formula, frame_idx, Z, R[Å], E[eV], F[eV/Å], role).
    role in {'reactant','product','ts','path'}; frame_idx=-1 for R/P/TS sub-groups.
    n_path: number of distinct interior path frames sampled per reaction (None = all)."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        grp = f[split]
        for formula in (formulas or sorted(grp.keys())):
            for rxn, g in grp[formula].items():
                Z = g["atomic_numbers"][()].astype(np.int64)
                special = {}
                for role, key in (("reactant", "reactant"), ("ts", "transition_state"), ("product", "product")):
                    s = g[key]
                    special[role] = (s["positions"][0], float(s[KE][0]), s[KF][0])
                if include_endpoints:
                    for role, (R, E, F) in special.items():
                        yield dict(reaction_id=rxn, formula=formula, frame_idx=-1,
                                   Z=Z, R=R, E=E, F=F, role=role)
                Rall = g["positions"]; n = Rall.shape[0]
                # drop path frames that duplicate R/P/TS (tolerance 1e-6 Å)
                Rnp = Rall[()]
                dup = np.zeros(n, bool)
                for (R, _, _) in special.values():
                    dup |= np.abs(Rnp - R[None]).max(axis=(1, 2)) < 1e-6
                idx = np.flatnonzero(~dup)
                if n_path is not None and len(idx) > n_path:
                    idx = np.sort(rng.choice(idx, n_path, replace=False))
                E_all = g[KE][()]; F_all = g[KF]
                for i in idx:
                    yield dict(reaction_id=rxn, formula=formula, frame_idx=int(i), Z=Z,
                               R=Rnp[i], E=float(E_all[i]), F=F_all[i], role="path")

# usage
# frames = list(iter_t1x("/content/transition1x.h5", "test", n_path=8))
# from ase import Atoms; atoms = Atoms(numbers=fr["Z"], positions=fr["R"])  # no cell, pbc=False
```

Count sanity check: `sum(len(f["test"][k]) for k in f["test"])` gives the number of test reactions.

---

## B. Training-data leakage check

| Model / checkpoint | Training data | Contains T1x? | Verdict |
|---|---|---|---|
| **MACE-MPA-0** (`mace_mp(model="medium-mpa-0")`) | MPtrj + sAlex (bulk crystals), PBE/PBE+U (VASP) per ACEsuit/mace-mp README | No. No isolated molecules at all | **Clean** (and strongly OOD) |
| **AIMNet2 `aimnet2`** (= `aimnet2-wb97m-d3_0`) | about 20M ωB97M-D3/def2-TZVPP structures (Anstine et al., Chem Sci 2025). Sources reported include ANI-1x/2x, OrbNet Denali, ChEMBL, peptide dimers | No T1x or TS dataset in the sources we could find **[SI table not fully read]** | **Clean for TS/path frames.** Composition-level overlap with ANI-1x (GDB-derived C/H/N/O ≤8 heavy atoms) is likely for near-equilibrium reactants and products |
| **AIMNet2 `aimnet2-rxn`** | 4,685,165 geometries: "AIMNet2 base set + RGD-1 reaction database, all recomputed at ωB97M-V/def2-TZVPP" (HF model card) | No T1x. RGD1 (Zhao/Savoie, YARP-enumerated C/H/N/O ≤10 heavy-atom reactions) is the **same chemical domain**, including TSs. The card reports a held-out "Grambow benchmark" (the parent set of T1x) at about 3 kcal/mol RMSD | **No direct leakage, but in-domain.** It is not an OOD test |
| **Orb `orb-v3-conservative-inf-omat`** | OMat24 (bulk inorganic, PBE/PBE+U) | No | **Clean** (OOD for molecules) |
| **Orb `orb-v3-conservative-omol` / `orbmol-v2`** | OMol25 (ωB97M-V/def2-TZVPD), plus OPoly26 for v2 | **YES.** OMol25 §2.5.1: "The Transition-1X dataset is a database of reactive trajectories and was recomputed in the UKS formalism." The Neutral split lists "ANI-2X, Orbnet Denali, SPICE2, GEOM, Transition-1X, and RGD1". The OOD reactivity test uses Grambow reactions "not found in either Transition-1x or RGD1" | **Configuration leakage.** Same geometries, relabelled at a different level of theory. Which T1x frames or formulas were subsampled is not stated |

**Recommended committee for the headline result:**

1. `mace_mp(model="medium-mpa-0")` is clean.
2. `AIMNet2Calculator("aimnet2")` (wB97M-D3 member 0) is clean and molecular, though TS geometries are off-distribution for it.
3. `orb-v3-conservative-inf-omat` is clean.

Run the same pipeline again with `aimnet2-rxn` and `orb-v3-conservative-omol` as a **sensitivity / "in-domain" arm**. That arm shows what happens when a member has seen reactive data.

**Caveats for the slide (verbatim-ready):**

- "Orb-omol / OrbMol were trained on OMol25, which includes Transition1x geometries relabelled at ωB97M-V. We exclude them from the headline numbers."
- "AIMNet2-rxn is trained on RGD1 reactions (no T1x), so it is in-domain for reactive chemistry. AIMNet2 (base) is the reported member."
- "No model is trained at ωB97x/6-31G(d). Every reported 'error' includes the level-of-theory gap (PBE or ωB97M vs ωB97x), not just ML error."
- "Evaluation uses the official T1x formula-held-out test split (8 formulas). Our oracle-labelled fine-tuning set comes from train formulas only."
- "Leakage checks rely on published dataset descriptions, not on geometry hashing against OMol25."

---

## C. Orb models (`orb-models` 0.7.0)

Registry `ORB_PRETRAINED_MODELS` (from `orb_models/forcefield/pretrained.py`):

| Name | Forces | Domain / data | Level of theory | Charge/spin input |
|---|---|---|---|---|
| `orbmol-v2` | conservative (+ learnable charges, Coulomb) | molecules, OMol25 + OPoly26 | ωB97M-V/def2-TZVPD | **required** |
| `orb-v3-conservative-omol` (alias `orbmol-v1-conservative`) | conservative | molecules, OMol25 | ωB97M-V/def2-TZVPD | **required** |
| `orb-v3-direct-omol` (alias `orbmol-v1-direct`) | direct (not -∇E) | molecules, OMol25 | ωB97M-V/def2-TZVPD | **required** |
| `orb-v3-conservative-inf-omat`, `orb-v3-conservative-20-omat` | conservative | materials, OMat24 | PBE/PBE+U | no |
| `orb-v3-direct-inf-omat`, `orb-v3-direct-20-omat` | direct | materials, OMat24 | PBE/PBE+U | no |
| `orb-v3-{conservative,direct}-{inf,20}-mpa` | as named | materials, MPtrj+Alexandria | PBE/PBE+U | no |
| `orb-v2`, `orb-d3-v2`, `orb-d3-{sm,xs}-v2`, `orb-mptraj-only-v2` | direct | materials | PBE(+D3) | no |

- `inf` means all neighbours within 6 Å (adapter `max_num_neighbors=120`). `20` caps neighbours at 20, which makes the PES discontinuous. **Use `inf` + `conservative` for forces.**
- Torch: `torch>=2.8.0,<3.0.0` and `requires-python>=3.12`, so **torch 2.11 is OK**. Other dependencies: `nvalchemi-toolkit-ops[torch]>=0.4.1,<0.5` (warp-lang; GPU kNN graph build) and `dm-tree==0.1.8`, which is pinned. "Windows support is not guaranteed". That is fine on Colab Linux.
- Models are `torch.compile`d by default on CUDA (dynamic shapes). Pass `compile=False` if torch 2.11/CUDA 13 compile misbehaves. `predict()["energy"]` is fp64 by default.
- v3 models also expose a per-atom **confidence head** (`calc.results["confidence"]`, 50 bins over 0–0.4 Å force MAE). Use it as a free single-model UQ baseline.

```python
import torch, numpy as np
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator

dev = "cuda"
orbff, adapter = pretrained.orb_v3_conservative_inf_omat(device=dev, precision="float32-highest", compile=False)
# molecule model instead:  orbff, adapter = pretrained.orb_v3_conservative_omol(device=dev, precision="float32-highest", compile=False)

# --- ASE single molecule (non-periodic: no cell, pbc=False) ---
atoms = Atoms(numbers=Z, positions=R)
atoms.info["charge"] = 0; atoms.info["spin"] = 1     # REQUIRED for *omol / orbmol*; must be set on ALL or NONE in a batch
atoms.calc = ORBCalculator(orbff, atoms_adapter=adapter, device=dev)
E, F = atoms.get_potential_energy(), atoms.get_forces()

# --- batched: many molecules, one forward pass ---
def orb_batch(atoms_list):
    g = adapter.from_ase_atoms_list(atoms_list, device=dev)      # GPU (alchemi) graph build
    out = orbff.predict(g, split=False)                          # conservative: forces = -dE/dR via autograd
    E = out["energy"].detach().cpu().numpy()                     # (B,) eV
    F = out["forces"].detach().cpu().numpy()                     # (sum N_i, 3) eV/Å
    F_list = np.split(F, np.cumsum(g.n_node.cpu().numpy())[:-1])
    return E, F_list
```

---

## D. AIMNet2 (`aimnet` 0.2.5, github.com/isayevlab/aimnetcentral)

- Install: `pip install "aimnet[ase,hf]"`. Torch `>=2.10` works with 2.11. Python `>=3.11`. Also needs `nvalchemi-toolkit-ops>=0.4.1` and `warp-lang>=1.13,<2`, which is compatible with Orb's `<0.5` pin, so pip resolves 0.4.x.
- Checkpoints (registry `aimnet/calculators/model_registry.yaml`). Each family has 4 ensemble members `_0.._3`. The short alias loads member 0.

| Alias | Canonical | Elements | Level of theory | Notes |
|---|---|---|---|---|
| `aimnet2` / `aimnet2-wb97m` | `aimnet2-wb97m-d3_0` | H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I | ωB97M-D3(BJ)/def2-TZVPP | general organic, charged OK |
| `aimnet2-b973c`, `aimnet2-2025` | `aimnet2-b973c[-2025]-d3_0` | same | B97-3c | |
| `aimnet2-nse` | `aimnet2-nse_0` | same | open-shell / radicals | takes `mult` |
| `aimnet2-pd` | `aimnet2-pd_0` | + Pd, −As | B97-3c/CPCM(THF) | |
| `aimnet2-rxn` | `aimnet2-rxn_0` | **H,C,N,O only** | ωB97M-V/def2-TZVPP | **neutral only** (`charge!=0` raises). Energies are on a shifted per-element scale, so use only differences |

- `aimnet2-rxn` dispersion is inconsistent between sources. The GCS registry adds D3(BJ) post hoc (`s8=0.3908,a1=0.566,a2=3.128`), while the HF card (`isayevlab/aimnet2-rxn`) says no D3 should be added. Pick one loader and keep it fixed.
- Charge input: `"charge"` is a float with shape `(B,)`. `"mult"` is used only by NSE.
- Batching modes:
  - **mode 0:** dense `(B,N,3)` with `numbers (B,N)`. Used on CUDA when N ≤ `nb_threshold=120`. All frames of one T1x reaction share Z, so this is ideal per reaction.
  - **mode 1:** flat `(ΣN,3)` + `mol_idx (ΣN,)`, with neighbour lists. Use it to mix reactions of different size.

```python
import torch, numpy as np
from aimnet.calculators import AIMNet2Calculator
calc = AIMNet2Calculator("aimnet2", device="cuda")            # or "aimnet2-rxn", "aimnet2-wb97m-d3_2", "isayevlab/aimnet2-rxn"

# (a) one reaction, B frames, same atoms -> dense mode 0
R = torch.tensor(np.stack([fr["R"] for fr in frames_of_rxn]), dtype=torch.float32, device="cuda")   # (B,N,3)
Zt = torch.tensor(frames_of_rxn[0]["Z"], device="cuda").expand(R.shape[0], -1)                      # (B,N)
out = calc({"coord": R, "numbers": Zt, "charge": torch.zeros(R.shape[0], device="cuda")}, forces=True)
E, F = out["energy"].detach().cpu().numpy(), out["forces"].detach().cpu().numpy()   # (B,), (B,N,3)

# (b) heterogeneous batch -> flat mode 1 with mol_idx
def aimnet_flat(frames):
    coord = np.concatenate([f["R"] for f in frames]); numbers = np.concatenate([f["Z"] for f in frames])
    mol_idx = np.concatenate([np.full(len(f["Z"]), i) for i, f in enumerate(frames)])
    out = calc({"coord": torch.tensor(coord, dtype=torch.float32), "numbers": torch.tensor(numbers),
                "charge": torch.zeros(len(frames)), "mol_idx": torch.tensor(mol_idx)}, forces=True)
    F = out["forces"].detach().cpu().numpy()
    return out["energy"].detach().cpu().numpy(), np.split(F, np.cumsum([len(f["Z"]) for f in frames])[:-1])
```

---

## E. MACE-MPA-0, MACE-OFF, seed-ensemble baseline

- **MACE-MPA-0:** trained on MPtrj + sAlex (3.5M crystals), PBE/PBE+U, 89 elements, MIT licence. Load it with `mace_mp(model="medium-mpa-0")`. That is also the **default** when `model=None` in mace-torch ≥0.3.10. `default_dtype="float32"` by default; pass `"float64"` for eval. Dependencies: `torch>=2.5`, `e3nn==0.4.4`.
- **Known failure modes on isolated organics / TS:**
  1. **Systematic PES softening.** uMLIPs trained on near-equilibrium MPtrj underestimate forces and curvature (Deng et al., npj Comput Mater 11, 9 (2025), arXiv:2405.07105).
  2. **No molecules or reactive TS geometries in training.** Molecular behaviour is shown only qualitatively in the MACE-MP-0 paper (Batatia et al., arXiv:2401.00096).
  3. **PBE itself underestimates barrier heights** by several kcal/mol relative to hybrids. Even a perfect PBE model has a systematic barrier gap to ωB97x. Pretrained MLIPs largely fail TS search on T1x: Zhao et al. Adv Sci 2025 found pretrained MACE-OFF23 located the intended TS for **51/960** T1x reactions and Orb-v2 for about 2%.
- **MACE-OFF23** (`from mace.calculators import mace_off; mace_off(model="small"|"medium"|"large")`): trained on SPICE at ωB97M-D3(BJ)/def2-TZVPPD, organic elements H,C,N,O,F,P,S,Cl,Br,I, ASL licence. No T1x/TS data **[per MACE-OFF paper; SPICE has no reaction paths]**. Also available: `mace_omol(model="extra_large")`, but it is trained on OMol25 and therefore T1x-contaminated. Do not use it in the headline.

Batched MACE (same path as `MACECalculator._atoms_to_batch`):

```python
import torch, numpy as np
from mace.calculators import mace_mp
from mace import data as mdata
from mace.tools import torch_geometric, utils as mutils
calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float64")
model = calc.models[0]; z_table = mutils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
heads = getattr(model, "heads", ["Default"]); keyspec = mdata.KeySpecification()
def mace_batch(atoms_list, bs=64):
    ds = [mdata.AtomicData.from_config(mdata.config_from_atoms(a, key_specification=keyspec, head_name=heads[-1]),
                                       z_table=z_table, cutoff=float(model.r_max), heads=heads) for a in atoms_list]
    Es, Fs = [], []
    for b in torch_geometric.dataloader.DataLoader(ds, batch_size=bs, shuffle=False):
        b = b.to("cuda"); out = model(b.to_dict(), compute_force=True, compute_stress=False, training=False)
        Es.append(out["energy"].detach().cpu().numpy()); F = out["forces"].detach().cpu().numpy()
        Fs += np.split(F, np.cumsum(np.bincount(b.batch.cpu().numpy()))[:-1])
    return np.concatenate(Es), Fs
```

### Seed-ensemble baseline: recommendation

**Primary (cheapest, defensible): K=4 fine-tunes of the worst model (MACE-MPA-0) with seeds 0–3.**

- Use the **identical** oracle-labelled subset and the same hyperparameters as the Fittest fine-tune. Vary only the seed: data order, plus LoRA A-init or head re-init.
- Compute UQ as σ_F, the per-structure std of forces across the 4 members.
- This is exactly the configuration the claim is about ("ensembles of one fine-tuned model collapse due to shared weights").
- Cost is 4× one LoRA run, typically minutes each on an L4.
- Caveat: LoRA initialises B=0, so all members start *identical* to the base model. Diversity then comes only from A-init and data order, which can look like a strawman. **Also report a "readout/full-fine-tune" variant**, in which the last interaction and readout layers are unfrozen with different seeds. Bilbrey et al. 2025 used this readout-ensembling construction.

**Free second baseline: AIMNet2's native 4-member ensemble** (`aimnet2-wb97m-d3_0..3`). These are independently trained on the same data, so they form a genuine deep ensemble at zero training cost. It is the strongest homogeneous competitor. If the heterogeneous committee beats it on AUROC, that is the headline result.

**Not recommended as primary: MACE-OFF small/medium/large.** The three sizes differ in both capacity and accuracy, so their disagreement mixes model bias with uncertainty, and K=3 gives no matched seeds.

**Supporting citations:**

- Kahle & Zipoli, PRE 105, 015311 (2022): NNP ensembles underestimate uncertainty.
- Bilbrey et al., npj Comput Mater 11, 109 (2025), doi:10.1038/s41524-025-01572-y: readout ensembles of foundation models.
- Liu et al., arXiv:2507.21297 (2025): "Shallow MACE ensembles can identify some OOD configurations yet systematically underestimate errors". The same paper finds that a heterogeneous ensemble of 11 uMLIPs gives Spearman ρ=0.87 against true force error.
- Tan et al., npj Comput Mater 9, 225 (2023).

---

## F. Energy referencing and "is this prediction wrong" metric

The models have different levels of theory and different absolute energy scales: MACE is VASP-PBE, AIMNet2-rxn uses a shifted scale, and Orb-omol is ωB97M-V. **Never compare absolute energies.** Use:

1. **Force error, primary.** Per-structure force error vs ωB97x is offset-free. For frame i:
   - `F_RMSE_i = sqrt(mean_{a,k}(F_pred - F_DFT)^2)`
   - `F_MAE_i = mean |ΔF|`
   - Disagreement signal: `D_i = mean_a || F_m,a - mean_m F_m,a ||` (committee spread, eV/Å), or the max over atoms for a per-atom flag.
2. **Relative energy within a reaction.** Composition is identical, so per-element offsets cancel:
   - `ΔE_i = E_i − E_reactant` (model and DFT, each self-referenced)
   - `err_i = |ΔE_i^pred − ΔE_i^DFT|`
   - Barrier error = `err` at `role=='ts'`.

```python
def rel_E(E_frames, E_reactant): return E_frames - E_reactant          # per model, per reaction
def frame_errors(Fp, Fd): d = Fp - Fd; return np.sqrt((d**2).mean()), np.abs(d).mean()
def committee_spread(F_models):  # list of (N,3)
    S = np.stack(F_models); return np.linalg.norm(S - S.mean(0), axis=-1).mean()
```

**Thresholds for the binary "wrong" label.** Report both an absolute label and a threshold-free score.

- **Barrier / relative energy:** `> 1 kcal/mol = 0.0434 eV` is "chemical accuracy". Use **> 2 kcal/mol (0.087 eV)** for "wrong", because the level-of-theory gap alone is often ≥1 kcal/mol between hybrids and much larger for PBE.
- **Forces:** **per-structure F_MAE > 0.1 eV/Å** (≈2.3 kcal/mol/Å). This is about 2× the in-domain error of a T1x-trained PaiNN (F MAE 0.058 eV/Å) and below an ANI1x-trained PaiNN on T1x (0.179 eV/Å) (Schreiner et al., Sci Data 2022). A stricter reference is OC20's EFwT, which requires energy within 0.02 eV and max force error within 0.03 eV/Å (Chanussot et al., ACS Catal 2021).
- **Threshold-free:** report AUROC/AUPRC of D_i for the label, and Spearman ρ(D_i, F_RMSE_i). Use oracle-budget curves: the fraction of "wrong" frames caught with k DFT lookups, compared with random and with single-model UQ (the Orb confidence head and the AIMNet2 own-ensemble σ).
- **Level-of-theory floor:** before thresholding, estimate it. After fine-tuning on ωB97x labels, the residual error approximates the irreducible ML error. Before fine-tuning, part of every model's "error" is PBE/ωB97M vs ωB97x. State this on the slide.

**Reported numbers to quote:**

- PaiNN (Schreiner 2022, eV and eV/Å) on the T1x test set:
  - trained on T1x: E MAE 0.048, F MAE 0.058
  - trained on ANI1x: E MAE 0.226, F MAE 0.179
- NEB vs GSM TS energies: MAE 0.16 eV.
- AIMNet2-rxn (HF card): about 2 kcal/mol RMSD on diverse barrier heights and about 3 kcal/mol on Grambow. A separate study found a barrier MAE of 2.9 kcal/mol (arXiv:2507.10400).
- Pretrained MACE-OFF23 found 51/960 intended TSs on T1x, and Orb-v2 about 2% (Zhao et al. 2025).
- On 29 organic TS searches, OMol25-trained models reach 86–97% success, with MACE-OMol25 at 96.6% (arXiv:2604.00405).
- **[unverified]:** direct force MAE of MACE-MPA-0, Orb-v3, or AIMNet2 on the T1x test split has not been published as far as we found. Our numbers will be new.

---

## G. Citations for the pitch

**Foundation MLIPs fail silently off-distribution (TS / reactive):**

1. Zhao, Han, Zhang, Wang, Zhong, Cui, Yin, Cao, Jia, Duan. "Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models." *Adv. Sci.* 12, e06240 (2025). doi:10.1002/advs.202506240. Pretrained foundation MLIPs "frequently fall short in reliably localizing TSs", and "traditional energy and force metrics alone do not reliably predict TS search success".
2. Schreiner, Bhowmik, Vegge, Busk, Winther. "Transition1x – a dataset for building generalizable reactive machine learning potentials." *Sci. Data* 9, 779 (2022). doi:10.1038/s41597-022-01870-w. arXiv:2207.12858. Models trained on equilibrium data (ANI1x) are 3–4× worse on reaction paths.
3. Deng, Choi, Zhong, Riebesell, Bhethanabotla, Ceder et al. "Systematic softening in universal machine learning interatomic potentials." *npj Comput. Mater.* 11, 9 (2025). arXiv:2405.07105.

**Same-model / fine-tuned ensembles underestimate uncertainty:**

4. Kahle, Zipoli. "Quality of uncertainty estimates from neural network potential ensembles." *Phys. Rev. E* 105, 015311 (2022).
5. Bilbrey (Pope), Firoz, Lee, Choudhury. "Uncertainty quantification for neural network potential foundation models." *npj Comput. Mater.* 11, 109 (2025). doi:10.1038/s41524-025-01572-y.
6. Liu, Wei, Gao, Dey, Sluiter, Shuang. "Heterogeneous Ensemble Enables a Universal Uncertainty Metric for Atomistic Foundation Models." arXiv:2507.21297 (2025). This is the closest prior art to Fittest (a cross-architecture committee of uMLIPs on materials). Our novelty: molecules/TS, an oracle budget, and fine-tuning the worst member.

**Committee disagreement as an error signal / query-by-committee AL:**

7. Smith, Nebgen, Lubbers, Isayev, Roitberg. "Less is more: Sampling chemical space with active learning." *J. Chem. Phys.* 148, 241733 (2018). doi:10.1063/1.5023802. This is ANI-1x QBC.
8. Schran, Brezina, Marsalek. "Committee neural network potentials control generalization errors and enable active learning." *J. Chem. Phys.* 153, 104105 (2020). arXiv:2006.01541.
9. Tan, Urata, Goldman, Kulik. "Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles." *npj Comput. Mater.* 9, 225 (2023). arXiv:2305.01754.

**Model papers:**

- MACE-MP-0: Batatia et al., arXiv:2401.00096
- MACE-OFF: Kovács et al., JACS 2025, arXiv:2312.15211
- Orb-v3: Rhodes et al., arXiv:2504.06231
- OMol25: Levine, Shuaibi, Taylor, Spotte-Smith, Hasyim et al., arXiv:2505.08762
- AIMNet2: Anstine, Zubatyuk, Isayev, Chem. Sci. 2025, doi:10.1039/D4SC08572H
- AIMNet2-rxn: Anstine, Zhao, Zubatiuk, …, Savoie, Isayev, ChemRxiv 2025, doi:10.26434/chemrxiv-2025-hpdmg

---

## Colab install (one cell)

```bash
pip install -q "mace-torch==0.3.17" "orb-models==0.7.0" "aimnet[ase,hf]==0.2.5" h5py
python -c "import torch,mace,orb_models,aimnet;print(torch.__version__, torch.version.cuda)"   # confirm torch was NOT downgraded
```

None of the three packages caps torch below 2.11: orb `<3`, aimnet `>=2.10`, mace `>=2.5`. If pip tries to swap torch, add `--no-deps` for the package that triggers it and install its remaining dependencies by hand.
