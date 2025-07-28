from abc import ABC, abstractmethod
import numpy as np


class BaseNormalizer(ABC):
    """
    Abstract base class for normalizers. Defines some necessary methods to be considered a normalizer.
    """

    @abstractmethod
    def fit(self, target):
        """Fit the normalizer to an target image"""


    @abstractmethod
    def transform(self, I):
        """Transform an image to the target stain"""

    @staticmethod
    def from_string(normalizer_type, normalizer_target_path):
        """Factory method to create a normalizer instance based on a string"""
        if normalizer_type is None:
            return None
            
        if normalizer_type == "reinhard":
            from normalizers.reinhard import ReinhardNormalizer
            return ReinhardNormalizer()
            
        elif normalizer_type == "macenko":
            from normalizers.macenko import MacenkoNormalizer
            return MacenkoNormalizer()

        elif normalizer_type == "vahadane":
            from normalizers.vahadane import VahadaneNormalizer
            return VahadaneNormalizer()
            
        elif normalizer_type == "staingan":
            from normalizers.staingan import StainGANNormalizer
            return StainGANNormalizer()
            
        elif normalizer_type == "tellez_augmentation":
            from normalizers.tellez_augmentation import TellezAugmentationNormalizer
            return TellezAugmentationNormalizer()
        
        elif normalizer_type == "multistain_cyclegan":
            from normalizers.multistain_cyclegan import MultistainCycleGANNormalizer
            return MultistainCycleGANNormalizer()
        
        elif normalizer_type == "histaugan":
            from normalizers.histaugan import HistAuGANNormalizer
            return HistAuGANNormalizer()

        elif normalizer_type == "torchvahadane":
            from normalizers.vahadane_torchvahadane import VahadaneNormalizer
            return VahadaneNormalizer()
            
        else:
            raise ValueError(f"Unknown normalization technique: {normalizer_type}")