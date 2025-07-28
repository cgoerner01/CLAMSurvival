from tiatoolbox.tools.stainnorm import VahadaneNormalizer as TTBVahadane
import numpy as np
from PIL import Image

from normalizers.base import BaseNormalizer

class VahadaneNormalizer(BaseNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = TTBVahadane()
    
    def fit(self, target_image):
        self.normalizer.fit(np.array(target_image))
        return

    def transform(self, image):
        return Image.fromarray(self.normalizer.transform(np.array(image)))