import einops
import torch
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid
from torch_cubic_spline_grids import CubicBSplineGrid3d

from .utils import array_to_grid_sample


def correct_motion(
    image: torch.Tensor,  # (t, h, w)
    deformation_field_data: torch.Tensor,  # (nt, nh, nw)
) -> torch.Tensor:
    t, h, w = image.shape
    deformation_field = CubicBSplineGrid3d.from_grid_data(deformation_field_data)
    grid = coordinate_grid(image_shape=image.shape) / torch.tensor([t - 1, h - 1, w - 1])
    predicted_shifts = deformation_field.forward(grid)
    sample_positions = grid - predicted_shifts
    corrected_image = F.grid_sample(
        input=einops.rearrange(image, 't h w -> t 1 h w'),
        grid=array_to_grid_sample(sample_positions, array_shape=(h, w)),
        mode='bicubic',
        padding_mode='zeros',
        align_corners=True,
    )
    return corrected_image
