import io
import typing

import numpy as np

from painting_estimation.images.utils import cv2_image_from_byte_io
from painting_estimation.inference.inference import EnsembleServing, ModelServing
from painting_estimation.model.effnet_model import SECOND_SERVING


SERVING: typing.Union[ModelServing, EnsembleServing] = SECOND_SERVING


def predict_painting_price(byte_io: io.BytesIO) -> float:
    rgb_numpy_image: np.ndarray = cv2_image_from_byte_io(byte_io=byte_io)
    price: np.ndarray | float = SERVING(rgb_numpy_image)

    if not isinstance(price, float):
        raise ValueError("Price estimation should be of type float")

    return price
