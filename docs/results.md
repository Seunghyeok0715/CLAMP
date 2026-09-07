# CLAMP results and comparison

## Source and interpretation

The values below are **author-reported paper results**, transcribed from [arXiv:2605.27990v1, Table 1 and Appendix C](https://arxiv.org/html/2605.27990v1#S3.T1). They are not new runs or independent replications. The selection includes all methods in Table 1 for three tasks, including an unfavorable case for CLAMP.

The protocol uses 100 validation images per dataset at 256 × 256, pixel-space diffusion priors, one NVIDIA RTX 6000 Ada Generation GPU, and batch size 1. Appendix C identifies the FFHQ prior from DPS and the ImageNet prior from Dhariwal & Nichol. Exact checkpoint hashes, image IDs, and raw per-image measurements are not supplied with this transcription. Release checkpoint links are in [assets.md](assets.md); their identity with the paper artifacts still requires verification.

CLAMP's budgets are T=50/K=5 for super-resolution, T=50/K=20 for nonlinear deblurring, and T=250/K=4 for phase retrieval. Its data-consistency scale is 0.01 and identity damping is 2.0. The release pixel presets generate measurements with noise standard deviation 0.05; Appendix C distinguishes physical measurement noise from solver calibration. Baseline budgets and method-specific settings are in Appendix C.2; these rows are not comparisons at equal iteration counts. Runtime is the paper's reported seconds; no new timing calibration was performed here.

## Selected pixel-space results

Higher PSNR/SSIM and lower LPIPS/runtime are better.

| Dataset | Task | Method | PSNR | SSIM | LPIPS | Time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FFHQ | 4× super-resolution | CLAMP | 29.515 | 0.841 | 0.219 | 6.743 |
| FFHQ | 4× super-resolution | DAPS | 28.619 | 0.764 | 0.262 | 28.257 |
| FFHQ | 4× super-resolution | SITCOM | 29.153 | 0.826 | 0.231 | 15.755 |
| FFHQ | 4× super-resolution | DMPlug | 28.637 | 0.797 | 0.253 | 118.255 |
| FFHQ | 4× super-resolution | DCDP | 27.611 | 0.785 | 0.225 | 4.719 |
| FFHQ | Nonlinear deblurring | CLAMP | 29.961 | 0.856 | 0.166 | 33.946 |
| FFHQ | Nonlinear deblurring | DAPS | 28.868 | 0.780 | 0.223 | 755.725 |
| FFHQ | Nonlinear deblurring | SITCOM | 29.519 | 0.812 | 0.207 | 27.167 |
| FFHQ | Nonlinear deblurring | DMPlug | 28.298 | 0.811 | 0.249 | 291.221 |
| FFHQ | Nonlinear deblurring | DCDP | 27.879 | 0.795 | 0.204 | 269.786 |
| ImageNet | Phase retrieval | CLAMP | 19.680 | 0.459 | 0.478 | 104.628 |
| ImageNet | Phase retrieval | DAPS | 22.354 | 0.519 | 0.402 | 241.113 |
| ImageNet | Phase retrieval | SITCOM | 18.481 | 0.383 | 0.524 | 94.187 |
| ImageNet | Phase retrieval | DCDP | 15.953 | 0.283 | 0.595 | 195.193 |

## Choosing baselines

CLAMP changes the likelihood correction using denoiser pullback, one-sided curvature, and aligned damping. DAPS uses decoupled noise annealing; SITCOM uses step-wise consistency optimization; DMPlug optimizes a diffusion input; DCDP separates data consistency and diffusion purification. See the paper's method section and Appendix C.2 for the compared implementations.

For FFHQ super-resolution, these rows show a favorable quality/runtime trade-off against DAPS, while DCDP runs faster. For FFHQ nonlinear deblurring, SITCOM runs faster than CLAMP. For ImageNet phase retrieval, DAPS has better PSNR, SSIM, and LPIPS at a longer runtime. Choose based on the required quality and compute budget, and compare under the same assets and evaluation protocol.

The paper's Table 3 contains latent comparisons against LatentDAPS, ReSample, and PSLD. Those numbers should be read separately from pixel results. For methods or settings absent from the paper, this release provides no direct comparison and makes no ranking claim. MRI results require a separate prior and data pipeline, which are not packaged here.
