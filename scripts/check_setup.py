"""Check the documented pixel quick start without loading checkpoint weights."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/ffhq256.pt')
    parser.add_argument('--data-root', default='dataset/test-ffhq')
    parser.add_argument('--imports-only', action='store_true',
                        help='Check Python dependencies only; no CUDA or assets required.')
    args = parser.parse_args()
    errors = []
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'PIL', 'yaml',
                 'piq', 'lpips', 'prettytable', 'wandb', 'setproctitle',
                 'tqdm', 'imageio', 'packaging'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    if not args.imports_only:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append('CUDA is unavailable. Install a CUDA-enabled torch wheel and compatible NVIDIA driver.')
            else:
                print(f'GPU: {torch.cuda.get_device_name(0)}; torch: {torch.__version__}')
        except ImportError:
            pass
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            errors.append(f'Missing or empty checkpoint: {checkpoint.resolve()}')
        root = Path(args.data_root)
        images = sorted(p for p in root.rglob('*')
                        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        if not images:
            errors.append(f'No PNG/JPEG images below {root.resolve()}; check the extracted archive layout.')
        else:
            print(f'Images: {len(images)}; first: {images[0]}')
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    if errors:
        return 1
    print('Setup checks passed.' if args.imports_only else
          'Setup checks passed. Run the README quick start to verify reconstruction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
