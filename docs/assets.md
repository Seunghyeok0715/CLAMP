# Checkpoints and datasets

Download only the prior and data needed for your experiment. These are the public links distributed with CLAMP; access and redistribution terms are controlled by the upstream providers. Model weights are not included under the repository MIT license.

For the first run, use the FFHQ DDPM and FFHQ data commands below. Inspect the extracted archive and set `--data_root` to the directory containing the images (recursive discovery is supported).

The archive commands below use Bash (`mkdir -p`, `wget`, `unzip`, `mv`). On Windows, use a Bash shell or extract ZIP files with `python -m zipfile -e ARCHIVE DESTINATION`. The Python inference command in the README also runs in PowerShell.

## Checkpoints


Create the checkpoint directory:

```bash
# From the CLAMP repository root
mkdir -p checkpoints
```

Required files (`./checkpoints`):

- `ffhq256.pt` (DDPM)
- `imagenet256.pt` (DDPM)
- `ldm_ffhq256.pt` (LDM)
- `ldm_imagenet256.pt` (LDM)
- `GOPRO_wVAE.pth` (nonlinear blur)

Download checkpoints:

- FFHQ DDPM:

```bash
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O checkpoints/ffhq256.pt
```

- ImageNet DDPM:

```bash
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O checkpoints/imagenet256.pt
```

- FFHQ LDM:

```bash
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./checkpoints
unzip checkpoints/ffhq.zip -d ./checkpoints
mv checkpoints/model.ckpt checkpoints/ldm_ffhq256.pt
rm checkpoints/ffhq.zip
```

- ImageNet LDM:

```bash
wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -P ./checkpoints/
mv checkpoints/model.ckpt checkpoints/ldm_imagenet256.pt
```

- Nonlinear blur:

```bash
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O checkpoints/GOPRO_wVAE.pth
```

## Datasets

Create the dataset directory:

```bash
# From the CLAMP repository root
mkdir -p dataset
```

Download datasets (the FFHQ archive contains `test-ffhq/` with 100 PNG images):

- FFHQ:

```bash
gdown https://drive.google.com/uc?id=1i0oI8nt_b9XCHNPKM5KR92Y4t8ZVMDvR -O dataset/test-ffhq.zip
unzip dataset/test-ffhq.zip -d ./dataset
rm dataset/test-ffhq.zip
```

- ImageNet:

```bash
gdown https://drive.google.com/uc?id=1ezXMhLt2UPaqNJnYNQAFM9ZLUW52ulz5 -O dataset/test-imagenet.zip
unzip dataset/test-imagenet.zip -d ./dataset
rm dataset/test-imagenet.zip
```
