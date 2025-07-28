from .base import BaseNormalizer
import numpy as np
import cv2 as cv
from PIL import Image

def standardize_brightness(I, percentile=95):
    """
    Standardize brightness.

    :param I: Image uint8 RGB.
    :return: Image uint8 RGB with standardized brightness.
    """
    assert is_uint8_image(I)
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0]
    p = np.percentile(L, percentile)
    I_LAB[:, :, 0] = np.clip(255. * L / p, 0, 255).astype(np.uint8)  # 255. float seems to be important...
    I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
    return I



def remove_zeros(I):
    """
    Remove zeros in an image, replace with 1's.

    :param I: An Array.
    :return: New array where 0's have been replaced with 1's.
    """
    mask = (I == 0)
    I[mask] = 1
    return I



def RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    I = remove_zeros(I)  # we don't want to take the log of zero..
    return -1 * np.log(I / 255)



def OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, 'Negative optical density'
    return (255 * np.exp(-1 * OD)).astype(np.uint8)



def normalize_rows(A):
    """
    Normalize the rows of an array.

    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]



def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'.
    Specifically, a pixel is not white if its luminance (in LAB color space) is less than the specified threshold.

    :param I: RGB uint 8 image.
    :param thresh: Luminosity threshold.
    :return: Binary mask where true denotes 'not white'.
    """
    assert is_uint8_image(I)
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)



def sign(x):
    """
    Returns the sign of x.

    :param x: A scalar x.
    :return: The sign of x  \in (+1, -1, 0).
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0



### Checks

def array_equal(A, B, eps=1e-9):
    """
    Are arrays A and B equal?

    :param A: Array.
    :param B: Array.
    :param eps: Tolerance.
    :return: True/False.
    """
    if A.ndim != B.ndim:
        return False
    if A.shape != B.shape:
        return False
    if np.mean(A - B) > eps:
        return False
    return True



def is_image(x):
    """
    Is x an image?
    i.e. numpy array of 2 or 3 dimensions.

    :param x: Input.
    :return: True/False.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True



def is_gray_image(x):
    """
    Is x a gray image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    squeezed = x.squeeze()
    if not squeezed.ndim == 2:
        return False
    return True



def is_uint8_image(x):
    """
    Is x a uint8 image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True



def check_image(x):
    """
    Check if is an image.
    If gray make sure it is 'squeezed' correctly.

    :param x: Input.
    :return: True/False.
    """
    assert is_image(x)
    if is_gray_image(x):
        x = x.squeeze()
    return x

class ReinhardNormalizer(BaseNormalizer):
    """
    Normalize a patch stain to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.

    staintools implementation, see https://staintools.readthedocs.io/en/latest/_modules/staintools/normalization/reinhard.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_means = None
        self.target_stds = None
        self.standardize = True

    def fit(self, target):
        """
        Fit to a target image

        :param target: Image RGB uint8.
        :return:
        """
        if self.standardize:
            target = standardize_brightness(target)
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds


    def transform(self, I):
        """
        Transform an image.

        :param I: Image RGB uint8.
        :return:
        """
        I = np.array(I)
        if self.standardize:
            I = standardize_brightness(I)
        I1, I2, I3 = self.lab_split(I)
        means, stds = self.get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return Image.fromarray(np.uint8(self.merge_back(norm1, norm2, norm3)))


    @staticmethod
    def lab_split(I):
        """
        Convert from RGB uint8 to LAB and split into channels.

        :param I: Image RGB uint8.
        :return:
        """
        assert is_uint8_image(I)
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I = I.astype(np.float32)
        I1, I2, I3 = cv.split(I)
        I1 /= 2.55
        I2 -= 128.0
        I3 -= 128.0
        return I1, I2, I3


    @staticmethod
    def merge_back(I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8.

        :param I1: L
        :param I2: A
        :param I3: B
        :return: Image RGB uint8.
        """
        I1 *= 2.55
        I2 += 128.0
        I3 += 128.0
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)


    def get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel.

        :param I: Image RGB uint8.
        :return:
        """
        I1, I2, I3 = self.lab_split(I)
        m1, sd1 = cv.meanStdDev(I1)
        m2, sd2 = cv.meanStdDev(I2)
        m3, sd3 = cv.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds