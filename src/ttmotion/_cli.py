from pathlib import Path
from typing import Optional

import einops
import mrcfile
import numpy as np
import typer

from .utils import imread
from .ttmotion import ttmotion as _ttmotion

cli = typer.Typer(no_args_is_help=True, add_completion=False)


@cli.command(no_args_is_help=True)
def ttmotion(
    input_file: Path = typer.Option(..., exists=True, file_okay=True),
    deformation_field_resolution: tuple[int, int, int] = typer.Option(..., ),
    output_average: Optional[Path] = typer.Option(...),
    output_average_even: Optional[Path] = typer.Option(...),
    output_average_odd: Optional[Path] = typer.Option(...),
    output_all_frames: Optional[Path] = typer.Option(...),
):
    image = imread(input_file)
    corrected_image, deformation_field_data = _ttmotion(
        image=image, deformation_field_resolution=deformation_field_resolution
    )

    if output_average is not None:
        average = einops.reduce(corrected_image, 't h w -> h w', reduction='mean')
        mrcfile.write(output_average, data=average.numpy().astype(np.float16))

    if output_average_even is not None:
        average = einops.reduce(corrected_image[::2], 't h w -> h w', reduction='mean')
        mrcfile.write(output_average_even, data=average.numpy().astype(np.float16))

    if output_average_odd is not None:
        average = einops.reduce(corrected_image[1::2], 't h w -> h w', reduction='mean')
        mrcfile.write(output_average_odd, data=average.numpy().astype(np.float16))

    if output_all_frames is not None:
        mrcfile.write(output_all_frames, data=corrected_image.numpy().astype(np.float16))
