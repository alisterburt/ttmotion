import torch

from ._patch_grid_centers import patch_grid_centers
from ._patch_grid_indices import patch_grid_indices


def patch_grid(
    images: torch.Tensor,
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(patch_shape) != len(patch_step):
        raise ValueError('patch shape and step must have the same number of dimensions.')
    ndim = len(patch_shape)
    if ndim == 2:
        patches, patch_centers = _patch_grid_2d(
            images=images,
            patch_shape=patch_shape,
            patch_step=patch_step,
            distribute_patches=distribute_patches,
        )
    elif ndim == 3:
        patches, patch_centers = _patch_grid_3d(
            images=images,
            patch_shape=patch_shape,
            patch_step=patch_step,
            distribute_patches=distribute_patches,
        )
    else:
        raise NotImplementedError()
    return patches, patch_centers


def _patch_grid_2d(
    images: torch.Tensor,
    patch_shape: tuple[int, int],
    patch_step: tuple[int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a grid of 2D patches from 2D image(s).

    Parameters
    ----------
    images: torch.Tensor
        `(..., h, w)` array of 2D images.
    patch_shape: tuple[int, int]
        `(patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).

    Returns
    -------
    patches, patch_centers: tuple[torch.Tensor, torch.Tensor]
        `(..., grid_h, grid_w, patch_h, patch_w)` grid of 2D patches
        and `(..., grid_h, grid_w, 2)` array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    patch_centers = patch_grid_centers(
        image_shape=images.shape[-2:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patch_idx_h, patch_idx_w = patch_grid_indices(
        image_shape=images.shape[-2:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patches = images[..., patch_idx_h, patch_idx_w]
    return patches, patch_centers


def _patch_grid_3d(
    images: torch.Tensor,
    patch_shape: tuple[int, int, int],
    patch_step: tuple[int, int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a grid of 3D patches from 3D image(s).

    Parameters
    ----------
    images: torch.Tensor
        `(..., h, w)` array of 3D images.
    patch_shape: tuple[int, int, int]
        `(patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).

    Returns
    -------
    patches, patch_centers: tuple[torch.Tensor, torch.Tensor]
        `(..., grid_d, grid_h, grid_w, patch_d, patch_h, patch_w)` grid of 2D patches
        and `(..., grid_h, grid_w, 2)` array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    patch_centers = patch_grid_centers(
        image_shape=images.shape[-3:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patch_idx_d, patch_idx_h, patch_idx_w = patch_grid_indices(
        image_shape=images.shape[-3:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patches = images[..., patch_idx_d, patch_idx_h, patch_idx_w]
    return patches, patch_centers
