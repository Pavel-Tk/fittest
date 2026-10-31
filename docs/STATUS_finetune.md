# STATUS: fine-tune agent (scripts/finetune.py)

## Working CLI (VM, from /content/fittest)
```
python3 scripts/finetune.py --frames data/frames.pkl --labelled-uids results/oracle_labels_armD.json \
  --model mace_mpa0 --out data/preds/mace_lora_curated.npz --ckpt data/ckpt/mace_lora_curated \
  --epochs 40 --seed 0 --predict-uids all
# seed-ensemble baseline (B'): same data and hold-out, seeds 0..3 -> data/preds/mace_seed{k}.npz, data/ckpt/mace_seed{k}/
python3 scripts/finetune.py ...same args... --seeds 0,1,2,3
```
`--labelled-uids` accepts a JSON list of uids, `[{"uid": ...}]` records, `{"uids"|"labelled_uids"|"labels"|"oracle_labels": ...}` wrappers, or a `{uid: ...}` map.

Before using the labels, the script re-derives splits with `assign_leakfree_splits(frames, --data-seed)`, the same call the ablation makes. Pass `--data-seed` equal to `run_ablation --seed` (default 0 for both). It then:
* refuses uids of split `eval` / `skeptic_ref` and uids without E/F;
* runs a geometry leak check. A labelled frame is excluded if its own `data.geometry_hash` matches any eval frame, or if a reactant/product of its reaction matches an eval endpoint. The result is recorded as `geometry_leak_check` in meta and in acceptance_report.json.

## Real-label runs
* **v1 (189 oracle labels, before the leak-free re-split): FAILED acceptance and was correctly not shipped.** The toolkit ratio was 1.37 because energy/atom MAE regressed. Hold-out rel-E MAE went 0.179 -> 0.307.
* Final run with the leaky by-reaction hold-out (05:00 UTC): superseded, not reported.
* Final run with the component hold-out: pending (job `ftfinal2`).

## Hold-out (fixed after claims audit)
* The previous by-reaction hold-out was leaky. In the final run its 9 hold-out reactions each shared an endpoint geometry with training reactions (T1x reuses reactants).
* It is now **whole connected components** of labelled reactions linked by shared endpoint geometry (`data.reaction_components` / `geometry_hash`, the helpers behind `assign_leakfree_splits`). Components are taken in seeded order up to 15 % of labelled frames. If that leaves less than 10 %, the smallest component that lifts it to at least 10 % is added, keeping at least 50 % for training.
* Zero shared geometries between train and hold-out is **asserted**. The result is recorded as `holdout_leak_check` in meta and in acceptance_report.json.
* Final labels (186) form 16 components. The hold-out is 3 components: 9 reactions, 26 frames (14.0 %), 0 shared geometries.

## Acceptance gates (all must pass; each recorded with `source`, value, limit, pass/fail)
* (a) ALCHEMI `build_acceptance_report(max_from_scratch_ratio=1.0)` against base MACE-MPA-0 with the same refitted E0s. Note that this reads **energy-per-atom MAE, i.e. absolute energies (E0 transfer)**, plus force MAE. It does not read the relative energies we score.
* (b) hold-out force RMSE <= base.
* (c) hold-out reaction-centred rel-E MAE <= base. If no hold-out reaction has 2 or more frames, this counts as not measurable and fails.
* `verdict` = "pass" iff (a) and (b) and (c). That verdict sets `shipped` and is what ReplayFinetune reads.

## Dev runs only (NOT real oracle labels), L4, 2026-10-31. Labels = 300 random pool frames from 134 reactions
These were run before gates (b)/(c), the leak check and the leak-free re-split were added, so they are illustrative only. Hold-out = 15 % of reactions: 41 frames from 20 reactions. "before" = base MACE-MPA-0 with the same refitted E0s.

| hold-out metric | base | LoRA (seed 0) |
|---|---|---|
| force RMSE [eV/Å] | 0.621 | **0.331** |
| force MAE [eV/Å] | 0.423 | 0.183 |
| mean per-frame F-RMSE | 0.606 | 0.271 |
| frac frames F-RMSE > 0.25 | 1.00 | 0.41 |
| rel-E MAE, reaction-centred [eV] | 0.195 | 0.154 |
| ALCHEMI acceptance (`max_from_scratch_ratio` 1.0) | | PASS (worst ratio 0.73), **dev run** |

Train-set force RMSE goes 0.653 -> 0.146. Unlabelled pool/eval frames: mean F-RMSE 0.235 / 0.264 for the 4-seed mean (base ~0.6).
* Wall time: one fine-tune takes 75 s train (40 epochs, 640 steps, bs 16) and 78 s with the acceptance step. Predicting all 5632 frames takes 4–9 s (42k atoms/s with cuEq warm). The whole script takes about 2 min, including E0 refit and base evaluation. The `--seeds 0,1,2,3` run takes 354 s.
* Seed ensemble (B'): the 4 members are nearly identical. Hold-out F-RMSE is 0.3303–0.3323. Mean σ_F on unlabelled frames is 0.009 eV/Å against a mean error of 0.24 eV/Å (about 26× under-dispersed), though Spearman(σ, err) is 0.80. This is the "seed ensemble underestimates error" effect.
* Earlier dev run on whole reactions (30 reactions, hold-out only 4 reactions) generalised worse: hold-out F-RMSE went 0.73 -> 0.44–0.52. With so few distinct reactions the cross-reaction gap is large.
* Loss-weight sweep (that dev set, 30 epochs) as E:F (hold-out F-RMSE / rel-E MAE / ratio verdict):
  * 1:10 gives 0.44 / 0.56 / **FAIL**: energy/atom MAE regressed by 9 %.
  * 10:10 gives 0.49 / 0.44 / pass.
  * 30:10 gives 0.52 / 0.40 / pass.
  * Default is 10:10, with per-atom energy Huber and per-component force Huber, delta 0.01. In "wrong"-threshold units forces still weigh about 13× more than energies.

## Recipe
* LoRA rank 16, alpha 2, using the targets and extra-trainable params of ALCHEMI example 08. The e3nn module names were checked on MACE-MPA-0. Training uses AdamW at lr 1e-3, wd 1e-4, cosine schedule down to 5 %, batch 16, 40 epochs. Training runs in the e3nn layout (`enable_cueq=False`) so the merged model exports as plain mace-torch. Evaluation and prediction use cuEq when it is available.
* E0 refit uses `fit_atomic_reference_energies` (ωB97x − MACE residual, added to the checkpoint E0s), fitted on the training frames only.
* Acceptance: see the gates above. `evaluate_accuracy` compares base+E0 with the reloaded checkpoint on the same hold-out batches. There is no absolute `max_forces_mae` bar: the Curator ships "better than what it replaces".

## Checkpoint format (for `BatchedMACE(model_path=...)`): `fittest/mace_lora.py`, `fittest.mace_lora/1`
```
<ckpt>/model.pt               merged plain mace-torch module (e3nn, float32, LoRA folded in, E0s baked in)
<ckpt>/fittest.json        manifest: format, base_checkpoint, atomic_energies, training config
<ckpt>/atomic_energies.json   {Z: E0}
<ckpt>/manifest.json, strategy.json, models/ ...   ALCHEMI PEFT checkpoint (trainable-only) at the same root
<ckpt>/acceptance_report.json
```
* I verified that BatchedMACE loads both `<ckpt>` (PEFT branch) and `<ckpt>/model.pt`. Both reproduce the script's predictions to 1e-3 eV (the float32 ulp at 1e4 eV) and 5e-6 eV/Å. `fittest.mace_lora.load_mace_lora(<ckpt>, enable_cueq=True)` is the canonical loader.
* Never call `MACEWrapper.from_checkpoint(<local path>)`. In 0.3.0-rc it tries to download the path as a URL and caches it by basename, so files with the same name collide.

## Predictions meta (for ReplayFinetune)
`base_model`, `train_uids` (85 % of the labelled frames), `holdout_uids`, `acceptance_report.verdict` ("pass"/"fail"), `shipped`, `epochs`, `n_train`, `wall_seconds`, `atoms_per_second`, `weights` (ckpt path). Because the hold-out is excluded, ReplayFinetune's `train_overlap` will be about 0.85, not 1.0.

## Not done / caveats
* `--full-finetune-last-layers` trains products.* + readouts.* fully, lr 1e-4, and it works. On the same dev set it is worse than LoRA:
  * hold-out F-RMSE 0.621 -> 0.382 (LoRA reaches 0.331)
  * rel-E MAE 0.195 -> 0.239, so relative energies regress
  * it barely passes the bar (worst ratio 0.994)
  * fine-tune wall time 65 s

  LoRA stays the main path.
