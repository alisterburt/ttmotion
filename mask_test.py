from torch_shapes import circle
import napari


ph, pw = (512, 512)

c1 = circle(image_shape=(ph, pw), radius=ph / 2.75, smoothing_radius=ph / 8)
c2 = circle(image_shape=(ph, pw), radius=ph / 4, smoothing_radius=ph / 8)

viewer = napari.Viewer()
viewer.add_image(c1.numpy())
viewer.add_image(c2.numpy())
napari.run()