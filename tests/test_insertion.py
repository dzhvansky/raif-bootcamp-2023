from io import BytesIO

import numpy as np

from painting_estimation.images.insertion import insert_image
from painting_estimation.images.utils import cv2_image_from_byte_io


def test_insertion(test_image: bytes) -> None:
    image = cv2_image_from_byte_io(BytesIO(test_image))
    inserted_img = insert_image(image, image)
    assert inserted_img.shape == image.shape
    assert inserted_img.dtype == image.dtype
    assert np.sum(np.all(inserted_img - image)) / np.sum(image) <= 0.15
