from normalizers.base import BaseNormalizer

class MacenkoNormalizer(BaseNormalizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, target_image):
        pass

    def transform(self, image):
        pass