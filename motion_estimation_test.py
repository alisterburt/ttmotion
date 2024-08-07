import mrcfile
import torch
import tifffile

from ttmotion.estimate_motion import estimate_motion


IMAGE_FILE = '/Users/burta2/data/EMPIAR-10491-5TS/frames/2Dvs3D_53-1_00001_-0.0_Jul31_10.36.03.tif'

# image = torch.tensor(tifffile.imread(IMAGE_FILE))
image = torch.rand(size=(12, 4096, 4096))
image = (image - torch.mean(image)) / torch.std(image)


estimate_motion(
    image=image,
    deformation_field_resolution=(5, 1, 1),
)