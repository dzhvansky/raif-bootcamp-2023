import numpy as np
from io import BytesIO
from painting_estimation.images.utils import ImgSize, image_size
from painting_estimation.images.preprocessing import ImagePreprocessor, cv2_image_from_byte_io


def test_preproc(test_image: bytes) -> None:
    preproc = ImagePreprocessor(
        target_size=ImgSize(width=100, height=200), target_dim_order=(0, 1, 2), target_dtype=np.float32
    )
    image = cv2_image_from_byte_io(BytesIO(test_image))
    img_preproc = preproc(image)
    assert img_preproc.dtype == np.float32
    assert ImgSize(img_preproc.shape[2], img_preproc.shape[1]) == ImgSize(width=100, height=200)
