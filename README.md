# CLAMP: Training-free diffusion inverse solver

**ICML 2026**

**Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping**

Seunghyeok Shin · Minwoo Kim · Dabin Kim · Hongki Lim
Inha University

[![arXiv](https://img.shields.io/badge/arXiv-2605.27990-b31b1b.svg)](https://arxiv.org/abs/2605.27990)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-4b44ce.svg)](https://openreview.net/forum?id=x9Cy1wydfo)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b6bfb.svg)](https://icml.cc/virtual/2026/poster/60728)

CLAMP (**Curvature-aware Langevin with Aligned Manifold Pullback**) reconstructs images from measurements using a pretrained diffusion prior, without training a task-specific solver. It replaces scalar likelihood guidance with denoiser-pullback curvature correction, manifold-aligned damping, and a matrix-free GMRES solve. Consider CLAMP as a baseline when comparing reconstruction quality and runtime for diffusion-prior inverse problems.

[Setup and first reconstruction](#installation) · [Checkpoints and datasets](docs/assets.md) · [Results and comparison](docs/results.md) · [Reproduction guide](docs/reproduction.md) · [Citation](#citation)

## Applicability and public implementation

| Component | Public code and presets |
| --- | --- |
| Images | FFHQ and ImageNet, 256 × 256 RGB |
| Pixel prior | DDPM, `--task_group pixel`, `--sampler edm_daps` |
| Latent prior | LDM, `--task_group ldm`, `--sampler latent_edm_daps`; additional dependencies required |
| Linear tasks | `down_sampling` (4×), `inpainting`, `inpainting_rand`, `gaussian_blur`, `motion_blur` |
| Nonlinear tasks | `phase_retrieval`, `nonlinear_blur`, `hdr` |
| Hardware | CUDA-enabled NVIDIA GPU; the main entry point requires CUDA |
| Additional assets | Pretrained prior and input images; nonlinear blur also needs the BKSE checkpoint |

A custom operator must provide the Jacobian actions required by `torch.func` (the default path uses automatic differentiation). Latent reconstruction also differentiates through the decoder. Training-free refers to the inverse solver; pretrained priors are still required.

**Paper scope versus release scope:** the paper also reports accelerated MRI and higher-resolution experiments. This repository does not supply a complete MRI dataset/prior/reconstruction recipe. Stable Diffusion wrappers are present, but are outside the documented 256 × 256 benchmark path. Availability of a preset does not mean that it has been independently reproduced; see the [validation status](docs/reproduction.md).

## Method

![CLAMP reconstructions on the inverse problems evaluated in the paper](assets/figure1.jpg)

![Denoiser pullback, curvature correction, and diffusion transition in CLAMP](assets/figure2.jpg)

At each noise level, CLAMP maps measurement sensitivity through the denoiser into the diffusion state. A one-sided curvature approximation and aligned rank-one damping define a correction solved with GMRES. A variance-preserving stochastic transition advances the noise schedule. In latent space, the measurement operator is composed with the decoder. See the [paper](https://arxiv.org/abs/2605.27990) for the derivation and assumptions.

## Installation

Run commands from the repository root. The validated quick-start stack is Python 3.11.9, PyTorch 2.3.0, torchvision 0.18.0, and CUDA 12.1 wheels. Install PyTorch from its explicit CUDA index before the remaining dependencies ([official wheel instructions](https://pytorch.org/get-started/previous-versions/#v230)).

```bash
git clone https://github.com/Seunghyeok0715/CLAMP.git
cd CLAMP
conda create -n clamp python=3.11.9 -y
conda activate clamp
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python main.py --help
```

`requirements.txt` covers the pixel-space path, evaluation, and download tools. The embedded LDM code additionally depends on legacy PyTorch Lightning, OmegaConf, einops, kornia, and CLIP; see [latent setup status](docs/reproduction.md#latent-and-mri-status) before using latent presets.

### Quick start: one FFHQ image, 4× super-resolution

Download the [FFHQ DDPM checkpoint and FFHQ test archive](docs/assets.md). The expected paths are `checkpoints/ffhq256.pt` and `dataset/test-ffhq/`. The singular directory name `dataset` matches the code presets. If your extracted archive has another layout, override `--data_root`.

```bash
python scripts/check_setup.py --checkpoint checkpoints/ffhq256.pt --data-root dataset/test-ffhq
python main.py --method clamp --task_group pixel --data test-ffhq --data_root dataset/test-ffhq --data_start_id 0 --data_end_id 1 --model ffhq256ddpm --sampler edm_daps --batch_size 1 --task down_sampling --name ffhq_sr4_smoke --anneal_num_steps 50 --clamp_gmres_iter 5 --clamp_lambda_id 2.0 --clamp_sigma_n 0.01 --operator_sigma 0.05 --seed 42 --save_traj false --eval_fn_list psnr,ssim --cudnn false
```

The first-run command sets `--cudnn false`, a native CUDA convolution fallback validated on Windows/RTX 4090. It still runs on the GPU. The default is `--cudnn true`; keep the backend setting in performance reports and validate it on your hardware.

This first run uses PSNR/SSIM to avoid downloading LPIPS's VGG weights. Add `--eval_fn_list psnr,ssim,lpips` for the benchmark metrics; LPIPS downloads its pretrained backbone on first use. Weights & Biases logging is disabled by default.

Expected files under `results/ffhq_sr4_smoke/`:

| File | Contents |
| --- | --- |
| `samples/00000_run0000.png` | Reconstructed RGB image |
| `grid_results.png` | Ground truth, visualized measurement, and reconstruction |
| `config.yaml` | Resolved data, operator, prior, sampler, and CLI configuration |
| `eval.md`, `metrics.json` | Reconstruction metrics and recorded sampling statistics |

Use a fresh `--name` for each experiment because matching output names overwrite files. A successful one-image run checks installation and execution; it does not reproduce a 100-image paper average.

### Benchmark commands

[configs_clamp_cli.txt](configs_clamp_cli.txt) contains the task commands and solver budgets. For ImageNet, replace `test-ffhq` with `test-imagenet` and `ffhq256ddpm` with `imagenet256ddpm` (or `ffhq256ldm` with `imagenet256ldm`). Update `--data_root` when it is supplied explicitly.

`--operator_sigma` controls the physical noise added to measurements. `--clamp_sigma_n` controls the solver's data-consistency calibration; these are different parameters. Keep both in any reported experiment configuration.

## Evidence and limitations

The [results guide](docs/results.md) provides readable quality/runtime comparisons with links to the source paper. Results are labeled as **author-reported in the paper**, separately from local execution checks. There is no claim of universal superiority: the appropriate baseline depends on the task, prior, metrics, and compute budget.

## Acknowledgements

This implementation builds upon [DAPS](https://github.com/zhangbingliang2019/DAPS), the nonlinear blur operator from [BKSE](https://github.com/VinAIResearch/blur-kernel-space-exploring), and the motion blur operator from [motionblur](https://github.com/LeviBorodenko/motionblur). Existing third-party notices and licenses are retained.

## Citation

Please cite the **ICML 2026 conference paper** below. GitHub's **Cite this repository** uses the same preferred paper citation from [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{shin2026clamp,
  author    = {Seunghyeok Shin and Minwoo Kim and Dabin Kim and Hongki Lim},
  title     = {Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://icml.cc/virtual/2026/poster/60728}
}
```
