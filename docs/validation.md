# Validation record — 2026-09-07

These are local release checks, separate from the author-reported paper results.

## Environment and assets

- Windows, Python 3.11.9, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121.
- NVIDIA GeForce RTX 4090, 24 GB VRAM, driver 560.94.
- Fresh `.venv-release` environment installed with the pixel requirements.
- Public FFHQ DDPM checkpoint and FFHQ ZIP downloaded using the links in [assets.md](assets.md). The archive contains 100 PNG images.

The README and CI use Python 3.11.9 to match the locally validated interpreter. Ubuntu CPU checks are configured in the CI workflow; the GPU reconstruction described here was validated on Windows.

## Completed checks

| Check | Outcome |
| --- | --- |
| Fresh pixel dependency installation and `pip check` | Passed; no broken requirements |
| Preflight imports, CUDA, checkpoint presence, image discovery | Passed; 100 images found |
| `python main.py --help` | Passed |
| Image discovery and RGB conversion | Passed; no duplicate entries, grayscale/RGBA accepted |
| Empty directory/slice failure | Passed; actionable error |
| All 16 published CLI task recipes and the README quick start | Parsed and resolved; this does not execute all priors |
| Frozen-prior correction regression | Passed; the actual input VJP/JVP and GMRES correction agrees with the unfrozen toy prior |
| Citation metadata | Passed official CFF 1.2.0 JSON Schema validation |
| Paper result transcription | 14 numeric rows with explicit paper provenance in the results guide |

The four automated regression tests run with `python -m unittest discover -s tests -v`. The workflow in `.github/workflows/release-checks.yml` runs CPU checks and does not certify GPU reconstruction or benchmark metrics.

## GPU reconstruction status

The T=50/K=5 one-image FFHQ super-resolution run completed using native CUDA convolutions (`--cudnn false`), with seed 42, physical noise 0.05, and solver calibration 0.01. The reconstructed image and grid were inspected. This is a local execution check on `test-ffhq/00000.png`, not an independent reproduction of a paper average.

The default cuDNN-enabled path stalled during denoiser VJP/backward on this Windows setup; those attempts were interrupted. Disabling cuDNN completed the same task and budget. This isolates a usable backend configuration, not a general diagnosis for other machines. The backend option is explicit and the default remains enabled.

The successful run is summarized below. Internal sampling time is a diagnostic with the limitations described in [reproduction.md](reproduction.md), and must not be compared directly against the paper runtime table.

| One-image smoke result | Value |
| --- | ---: |
| PSNR | 31.1733 dB |
| SSIM | 0.8641 |
| Recorded sampling time | 26.273 seconds |
| Recorded peak allocated CUDA memory | 2.781 GiB |

The final run used the README CLI directly, including `--cudnn false`, and produced every documented output file. After validation, the downloaded checkpoints/dataset, temporary environment, local output images, and separate benchmark artifacts were removed from the workspace at the maintainer's request. This document retains the validation summary; raw run artifacts are not included in the release.

Full 100-image quality/runtime reproduction, LPIPS/FID, latent priors, ImageNet, nonlinear blur, and MRI remain unverified here. Their validation still requires the original experiment environment and complete artifact/protocol information.
