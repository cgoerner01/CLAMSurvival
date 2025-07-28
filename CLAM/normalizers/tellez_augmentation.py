import numpy as np
import random
from skimage.color import rgb2hed, hed2rgb
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from normalizers.base import BaseNormalizer

class HEDLightTransform(ImageOnlyTransform):
    def __init__(self, intensity_range=(-0.05, 0.05), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range

    def apply(self, image, **params):
        image = image.astype(np.float32) / 255.0
        hed = rgb2hed(image)

        for i in range(3):
            ratio = np.random.uniform(*self.intensity_range)
            hed[..., i] *= (1 + ratio)

        rgb = hed2rgb(hed)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)



class TellezAugmentationNormalizer(BaseNormalizer):
    """
    Color augmentation method described in Tellez et al. paper
    "Quantifying the effects of data augmentation and stain color normalization in 
    convolutional neural networks for computational pathology"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, target_image):
        print("TellezAugmentationNormalizer does not need to be fitted -> pass")
        pass
    
    def get_transform(self):
        elastic_transform_alpha = random.uniform(*(80,120))
        elastic_transform_sigma = random.uniform(*(9.0,11.0))

        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.ElasticTransform(
                alpha=elastic_transform_alpha, sigma=elastic_transform_sigma,
                border_mode=0, p=0.5
            ),

            A.RandomScale(scale_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.0, 0.1), p=0.5),
            A.GaussianBlur(sigma_limit=(0.0,0.1), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.5, p=0.5
            ),

            HEDLightTransform(p=1.0),  # custom stain perturbation

        ])


    def transform(self, image):
        img_np = np.array(image)
        transform = self.get_transform()
        augmented = transform(image=img_np)['image']

        return Image.fromarray(augmented)
    

