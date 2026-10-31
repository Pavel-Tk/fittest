# STATUS: fine-tune agent (scripts/finetune.py)

## Checkpoint format (for `BatchedMACE(model_path=...)`)
`trustswarm/mace_lora.py` is the single source of truth. Format `trustswarm.mace_lora/1`:

```
data/ckpt/mace_lora_curated/
  model.pt               merged plain mace-torch ScaleShiftMACE nn.Module, e3nn layout, float32,
                         LoRA folded in, T1x-refitted E0s baked in -> energies on the wB97x scale
  manifest.json          format, base_checkpoint "medium-mpa-0", atomic_energies {Z: E0}, training config
  peft/                  ALCHEMI PEFT checkpoint (FineTuningStrategy.save_checkpoint, trainable-only)
  acceptance_report.json ALCHEMI acceptance report + hold-out metrics
```
Load it with:
* plain torch: `torch.load(f"{model_path}/model.pt", map_location="cuda", weights_only=False)` (a mace-torch module, same API as the foundation `.model` files)
* ALCHEMI: `from trustswarm.mace_lora import load_mace_lora; m = load_mace_lora(model_path, device="cuda", enable_cueq=True)` returns a frozen `MACEWrapper`
* `model_path` may be the directory or the `model.pt` file (`mace_lora.resolve_model_file`).
* Do NOT use `MACEWrapper.from_checkpoint(<local path>)`. In 0.3.0-rc it sends local paths through `download_mace_mp_checkpoint`, which tries to download them as URLs and caches them by sanitised basename, so two different `model.pt` files collide in `~/.cache/mace`.

## Progress
* [x] API verified on the VM: MACE-MPA-0 e3nn module names match the ALCHEMI example LoRA patterns; `density_fn.layer0.weight` exists (extra trainable).
* [ ] first end-to-end run (dev set: 30 pool reactions, 330 frames)
