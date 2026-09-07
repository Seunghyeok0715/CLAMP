# Reproducing CLAMP

## First-run procedure

Follow the [README](../README.md#installation), download the [public assets](assets.md), run `scripts/check_setup.py`, then run the one-image command. The preflight checks imports, CUDA availability, checkpoint presence, and image discovery. It does not load weights, verify hashes, or prove reconstruction quality.

Use RGB PNG/JPEG files; the loader converts grayscale and RGBA images to RGB. Image paths are sorted, then sliced by `--data_start_id` and `--data_end_id` (exclusive). The loader resizes the short edge and center-crops to the preset resolution. Empty slices fail with an actionable error. Run from the repository root so relative checkpoint/operator paths resolve correctly.

If the denoiser backward pass stalls with cuDNN on Windows, use the documented `--cudnn false` fallback. This selects native CUDA convolutions and was used for the successful local smoke test. It can change speed and floating-point results; include the backend setting when reporting experiments. `--cudnn true` remains the default for existing commands.

## Benchmark protocol

For paper comparisons, keep the 100-image selection, pretrained checkpoint, measurement realization, seed, prior representation, task settings, and solver budgets fixed. Use [configs_clamp_cli.txt](../configs_clamp_cli.txt) for CLAMP task budgets. Save `config.yaml`, `metrics.json`, a full package freeze, checkpoint SHA-256, image-list identifiers, GPU/driver details, and the source commit with each run.

Set `--batch_size 1`. The current entry point places the selected image set on the GPU before sampling; reducing the sampling batch size does not reduce all input storage. Disable trajectories with `--save_traj false` for a first run. A 1-image smoke test is not a 100-image benchmark and should never be merged into paper-result CSVs.

The existing `average_time_per_sample_seconds` field averages sampling calls and only corresponds to per-image time at batch size 1. Internal timers do not explicitly synchronize CUDA boundaries and exclude some evaluation/output operations. Treat them as implementation diagnostics, not a new validated wall-clock comparison with the paper. Latent sampling currently does not report peak GPU memory.

## Latent and MRI status

Latent presets and embedded model sources are available. A tested standalone latent dependency lock and full checkpoint run remain to be established. The code imports legacy `pytorch_lightning.utilities.distributed`, OmegaConf, einops, kornia, and CLIP; installing the latest versions without a compatibility check is not a reproduction recipe. The published ImageNet preset is class-conditional with a placeholder label, while Appendix C describes an unconditional prior. Confirm the exact paper checkpoint and conditioning configuration before claiming latent reproduction.

MRI reproduction additionally needs the MRI prior, dataset access/preprocessing, sampling masks, and evaluation scripts. No complete MRI recipe is supplied by this public entry point.

## Validation status

See [validation.md](validation.md) for checks actually performed on this revision. Paper results remain in [results.md](results.md), with their own provenance.
