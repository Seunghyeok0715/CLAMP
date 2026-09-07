"""Release regressions that do not require model weights or a GPU."""
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from data import ImageDataset
from main import CLAMP, build_configs, parse_args


class ImageInputTests(unittest.TestCase):
    def test_unique_discovery_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            Image.new('RGB', (20, 20)).save(Path(root) / 'a.PNG')
            Image.new('RGBA', (20, 20)).save(Path(root) / 'b.png')
            Image.new('L', (20, 20)).save(Path(root) / 'c.jpg')
            dataset = ImageDataset(root=root, resolution=16, device='cpu')
            self.assertEqual(len(dataset), 3)
            for image in dataset:
                self.assertEqual(tuple(image.shape), (3, 16, 16))

    def test_empty_root_and_empty_slice_have_actionable_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, '--data_root'):
                ImageDataset(root=root, device='cpu')
            Image.new('RGB', (20, 20)).save(Path(root) / 'a.png')
            with self.assertRaisesRegex(ValueError, '--data_root'):
                ImageDataset(root=root, start_id=1, end_id=1, device='cpu')


class RecipeTests(unittest.TestCase):
    def test_all_published_task_commands_resolve(self):
        commands = Path('configs_clamp_cli.txt').read_text().splitlines()
        commands += Path('README.md').read_text(encoding='utf-8-sig').splitlines()
        for line in commands:
            if not line.startswith('python main.py --method '):
                continue
            with self.subTest(command=line), patch.object(sys, 'argv', shlex.split(line)[1:]):
                args = parse_args()
                configs = build_configs(args)
                self.assertEqual(configs['sampler_cfg']['latent'], args.task_group == 'ldm')
                self.assertEqual(configs['operator_cfg']['sigma'], 0.05)


class FrozenPriorTests(unittest.TestCase):
    def test_input_pullback_correction_is_preserved_with_frozen_weights(self):
        # Exercise the actual correction, including input VJP/JVP and GMRES.
        prior = torch.nn.Conv2d(3, 3, 1, bias=False)
        with torch.no_grad():
            prior.weight.copy_(0.8 * torch.eye(3).reshape(3, 3, 1, 1))
        solver = CLAMP({'name': 'edm', 'num_steps': 3, 'sigma_max': 1,
                        'sigma_min': 0.1, 'timestep': 'poly-7'})
        solver.A = torch.nn.Identity()
        x = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
        y = x * 0.5

        def evaluate(z):
            denoised = prior(z)
            return denoised, (z - denoised) / 0.5

        enabled = solver._penalized_gn_step(x_ref=x, y=y, sigma_prior=0.5,
                                           x_eval_fn=evaluate)[0]
        prior.requires_grad_(False)
        frozen = solver._penalized_gn_step(x_ref=x, y=y, sigma_prior=0.5,
                                          x_eval_fn=evaluate)[0]
        self.assertTrue(torch.isfinite(frozen).all())
        self.assertGreater(frozen.norm().item(), 0)
        torch.testing.assert_close(frozen, enabled)


if __name__ == '__main__':
    unittest.main()
