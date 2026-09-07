# CLAMP v0.1.0

CLAMP is the training-free diffusion inverse solver introduced at ICML 2026 in *Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping*.

This initial research-code release includes a verified FFHQ pixel-space quick start. Full paper benchmark reproduction, latent-space setup, and MRI reproduction remain unverified. See the [validation record](https://github.com/Seunghyeok0715/CLAMP/blob/v0.1.0/docs/validation.md) for the tested scope.

This update makes the public implementation easier to identify, cite, and run:

- Prominent method identity, author names, applicability, and paper/code scope in the README.
- GitHub citation metadata with the official paper title and author order.
- Pixel inference dependency manifest, corrected dataset directory instructions, preflight checks, and a one-image reconstruction command with documented outputs.
- Image discovery that avoids duplicate inputs on case-insensitive filesystems, RGB conversion, and an explicit empty-dataset error.
- Fixed malformed latent example names and added CPU CI for imports, data loading, and all published task commands.
- Added an explicit `--cudnn false` native CUDA fallback and verified a one-image FFHQ reconstruction on Windows/RTX 4090.
- Readable paper quality/runtime comparisons with source links, including limitations and cases favoring other methods.
- A reproduction guide and an explicit record of completed and pending validation.
- Installation instructions and CPU CI aligned with the validated Python 3.11.9 interpreter; tracked Python bytecode excluded from the source release.

Pretrained weights are frozen during inference while input Jacobian actions remain enabled. The correction equations are unchanged. Checkpoints and datasets remain [external downloads](https://github.com/Seunghyeok0715/CLAMP/blob/v0.1.0/docs/assets.md).

Use the [README quick start](https://github.com/Seunghyeok0715/CLAMP/blob/v0.1.0/README.md#installation) for the first reconstruction. The local Windows/RTX 4090 check used `--cudnn false`; the default cuDNN backend remains enabled for existing commands. Internal timing diagnostics should not be compared directly with the paper's reported runtimes.
