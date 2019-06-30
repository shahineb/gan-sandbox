from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
from functools import wraps


"""
Useful decorators for image processing
"""


def tocv2(func):
    """Ensures image fed to callable is cv2 compliant
    """
    @wraps(func)
    def wrapper(self, img, *args, **kwargs):
        if isinstance(img, PIL.Image.Image):
            img = imutil.PIL2cv2(img, BGR=False)
        elif not isinstance(img, np.ndarray):
            raise TypeError("Unkown image type")
        return func(self, img, *args, **kwargs)
    return wrapper


def toPIL(func):
    """Ensures image fed to callable is PIL compliant
    """
    @wraps(func)
    def wrapper(self, img, *args, **kwargs):
        if isinstance(img, np.ndarray):
            img = imutil.cv22PIL(img)
        elif not isinstance(img, PIL.Image.Image):
            raise TypeError("Unkown image type")
        return func(self, img, *args, **kwargs)
    return wrapper


def tight_render(func):
    """Rendering options when plotting an image with matplotlib
    """
    @wraps(func)
    def wrapper(self, img, *args, **kwargs):
        output = func(self, img, *args, **kwargs)
        plt.axis('off')
        plt.tight_layout()
        return output
    return wrapper

###############################################################################


class imutil:
    """Base class for image processing utilities
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def PIL2cv2(cls, pil_img, BGR=False):
        """Converts PIL format image to cv2

        Args:
            pil_img (PIL.Image)
            BGR (bool): if True, changes color format to BGR (PIL default)
        """
        cv2_img = np.array(pil_img)
        if len(cv2_img.shape) == 3 and cv2_img.shape[-1] > 3:
            cv2_img = cv2_img[:, :, :3]
        if BGR:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        return cv2_img

    @classmethod
    def cv22PIL(cls, cv2_img, BGR=False):
        """Converts cv2 image format to PIL

        Args:
            cv2_img (np.ndarray)
            BGR (bool): if True, assumes input image format is BGR
        """
        if BGR:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(cv2_img)
        return pil_img

    @classmethod
    @tocv2
    def cvtColor(cls, img, src, tgt):
        """Converts image color space with opencv convention

        Args:
            img (PIL.Image.Image, np.ndarray)
            src (str): source color space
            tgt (str): target color space eg {'RGB', 'BGR', 'GRAY', 'XYZ', 'YCrCb', 'HSV', 'HLS', 'Bayer'}

        Returns:
            output (np.ndarray)

        """
        attribute = "COLOR_" + src + "2" + tgt
        try:
            output = cv2.cvtColor(img, cv2.__getattribute__(attribute))
        except AttributeError:
            raise AttributeError(f"{attribute} is not a valid cv2 color conversion attribute. Please refer to opencv documentation")
        return output


###############################################################################


class Handler(imutil):
    # TODO : handle torch tensors (torchvision grid and simple tensors)

    COLORS = {'r': (0, 255, 0)}

    def __init__(self, plt):
        """Initializes the tiles handler to manipulate tiles images

        Args:
            plt: pyplot module used for display
        """
        self._plt = plt

    def load(self, img_path, enforce_cv2=False, enforce_PIL=False):
        """Returns img
        Args:
            img_path (str)
        """
        if img_path.endswith(".jpg"):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise FileNotFoundError(f"[Errno 2] No such file or directory: {img_path}")
            if enforce_PIL:
                img = Handler.cv22PIL(img)
        elif img_path.endswith(".png"):
            img = PIL.Image.open(img_path)
            if enforce_cv2:
                img = Handler.PIL2cv2(img)
        else:
            raise UnboundLocalError("Image processing librairy misunderstood")
        return img

    @tight_render
    def show(self, img):
        if isinstance(img, PIL.Image.Image):
            return img
        if isinstance(img, np.ndarray):
            return self._plt.imshow(img)

    @tocv2
    @tight_render
    def draw_boxes(self, img, bboxes, figsize=(10, 10), color='r', thickness=1,
                   fig=None, ax=None):
        """Draws bounding boxes on image

        Args:
            img (PIL.Image.Image, np.ndarray)
            bboxes (list): list of bounding boxes formatted as [x1, y1, x2, y2]
            figsize (tuple): figure dimensions
            color (str): boxes color (see Handler.COLORS)
            thickness (int): thickness of the boxes
        """
        if not ax:
            fig, ax = self._plt.subplots(figsize=figsize)
        ax.imshow(img)
        for bbox in bboxes:
            ax.add_patch(self._plt.Rectangle((bbox[0], bbox[1]),
                                             bbox[2] - bbox[0],
                                             bbox[3] - bbox[1],
                                             fill=False,
                                             edgecolor=color,
                                             linewidth=thickness))
        return fig, ax
