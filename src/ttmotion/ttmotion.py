import torch

from .estimate_motion import estimate_motion
from .correct_motion import correct_motion


def ttmotion(
    image: torch.Tensor,
    deformation_field_resolution: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    deformation_field_data = estimate_motion(
        image=image,
        deformation_field_resolution=deformation_field_resolution,
        patch_sidelength=(512, 512),
    )
    motion_corrected_frames = correct_motion(
        image=image,
        deformation_field_data=deformation_field_data
    )
    return motion_corrected_frames, deformation_field_data
