# Maintainer follow-up

## Materials needed for a stronger research release

- Original Python/conda package export, exact checkpoint hashes, and paper image-list identifiers, especially for the latent ImageNet prior/conditioning discrepancy recorded in the reproduction guide.
- Full benchmark CSVs with per-image metrics, runtime measurement boundaries, seeds, baseline commands/commits, and hardware metadata. The current results guide contains a paper-table transcription.
- MRI prior, data access/preprocessing instructions, masks, and runnable evaluation entry point if MRI is to be advertised as reproducible from this release.
- Add the paper-specific PMLR landing page, page range, and proceedings DOI if available. The preferred citation uses ICML 2026 / PMLR volume 306, as specified in the official ICML 2026 formatting instructions, and links to the official ICML paper page. The arXiv DOI is not used as a proceedings DOI.

## Public identity updates

Suggested GitHub About description: `CLAMP: training-free diffusion inverse solver with denoiser-pullback curvature and manifold-aligned damping (ICML 2026).`

Suggested topics: `inverse-problems`, `diffusion-models`, `posterior-sampling`, `image-reconstruction`, `icml-2026`.

For an author-managed arXiv revision, a minimal proposed abstract change is to replace “We replace scalar guidance” with “We introduce CLAMP (Curvature-aware Langevin with Aligned Manifold Pullback), which replaces scalar guidance”. Keep the official title and author order unchanged, and align manuscript and metadata through the normal submission process. This local file does not update arXiv.

A permanent project website can reuse the method identity, applicability, comparison data, and citation in this repository. Choose the author/lab-owned host and canonical URL before publishing and linking it. Crawler policy must be configured at that host; a robots.txt in this code repository does not configure github.com.
