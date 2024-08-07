import einops
import numpy as np
import torch
from torch_cubic_spline_grids import CubicBSplineGrid3d
from torch_shapes import circle
from torch_fourier_shift import fourier_shift_dft_2d

from .patch_grid import patch_grid


def estimate_motion(
    image: torch.Tensor,  # (t, h, w)
    deformation_field_resolution: tuple[int, int, int],  # (t, h, w)
    deformation_field_data: torch.Tensor | None = None,
    patch_sidelength: tuple[int, int] = (512, 512),
    n_iterations: int = 200,
    n_patches_per_batch: int = 20,
    learning_rate: float = 0.05,
) -> torch.Tensor:
    t, h, w = image.shape

    # initialise a parametrisation of the deformation field
    # deformation field is 2D shifts resolved over image dimensions (t, h, w)
    if deformation_field_data is not None:
        deformation_field = CubicBSplineGrid3d.from_grid_data(deformation_field_data)
    else:
        deformation_field = CubicBSplineGrid3d(
            resolution=deformation_field_resolution,
            n_channels=2,  # (dh, dw)
        )  # (dydyx, nt, nh, nw)

    # extract a grid of 2D patches covering the field of view
    ph, pw = patch_sidelength
    patches, patch_positions = patch_grid(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )  # (t, grid_h, grid_w, 1, ph, pw)
    t, gh, gw, _, ph, pw = patches.shape
    patches = einops.rearrange(patches, 't gh gw 1 ph pw -> t gh gw ph pw')
    patches = patches.detach()

    # normalise patch positions for evaluation of shifts on deformation field
    # which is parametrised over [0, 1]
    patch_positions = patch_positions / torch.tensor([t - 1, h - 1, w - 1])

    # initialise optimiser
    motion_optimiser = torch.optim.Adam(
        params=deformation_field.parameters(),
        lr=learning_rate
    )

    # mask data patches with a soft, circular mask
    mask = circle(radius=ph / 4, image_shape=(ph, pw), smoothing_radius=ph / 8)
    patches = patches * mask

    # calculate rfft2 of data, enables fast shift in fourier space
    patches = torch.fft.rfftn(patches, dim=(-2, -1))

    for i in range(n_iterations):
        # work on a subset of patches in each iteration
        idx_h, idx_w = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(2, n_patches_per_batch)
        )
        patch_subset = patches[:, idx_h, idx_w]
        patch_subset_positions = patch_positions[:, idx_h, idx_w]

        # query model for per-patch shifts
        predicted_shifts = deformation_field(patch_subset_positions)
        shifted_data_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts,
            rfft=True,
            fftshifted=False,
        )  # (t, b, ph, pw)

        # calculate reference as sum of shifted patches
        references = einops.reduce(shifted_data_patches, 't b ph pw -> b ph pw', reduction='sum')

        # calculate loss
        diff = torch.abs(references - shifted_data_patches) ** 2
        loss = torch.mean(diff)

        # update model parameters to minimise loss
        motion_optimiser.zero_grad()
        loss.backward()
        motion_optimiser.step()

        if i % 20 == 0:
            print(loss.item())
            print(deformation_field.data)
        return deformation_field.data.cpu().detach().numpy()
