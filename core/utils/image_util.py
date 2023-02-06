import os
import shutil

from termcolor import colored
from PIL import Image
import numpy as np
import imageio


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    return image_pil


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))


def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout

     
class ImageWriter():
    def __init__(self, output_dir, exp_name, fps=20):
        self.image_dir = os.path.join(output_dir, exp_name)

        print("The rendering is saved in " + \
              colored(self.image_dir, 'cyan'))
        
        # remove image dir if it exists
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        
        os.makedirs(self.image_dir, exist_ok=True)
        self.video_writer = imageio.get_writer(f'{self.image_dir}/video.mp4', mode='I', fps=fps, codec='libx264')
        self.frame_idx = -1
        self.frame_idx_np = -1

    def append(self, image, img_name=None):
        self.frame_idx += 1
        if img_name is None:
            img_name = f"{self.frame_idx:06d}"
        img = save_image(image, f'{self.image_dir}/{img_name}.png')
        self.video_writer.append_data(image)
        return self.frame_idx, img_name

    def append_numpy(self, data):
        self.frame_idx_np += 1
        file_name = f"{self.frame_idx_np:06d}"
        np.save(f'{self.image_dir}/{file_name}.npy', data)

    def finalize(self):
        self.video_writer.close()
