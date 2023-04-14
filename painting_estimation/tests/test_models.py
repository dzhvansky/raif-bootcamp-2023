import numpy as np

from painting_estimation.models import FirstModel


def white_image() -> np.ndarray:
    return np.full(fill_value=255, shape=(224, 224, 3), dtype=np.uint8)


def non_squared_image() -> np.ndarray:
    return np.full(fill_value=255, shape=(100, 224, 3), dtype=np.uint8)


def test_white_image() -> None:
    model = FirstModel()
    price = model.predict(white_image())
    assert np.isclose(price, 2247.627136400723)


def test_non_squared_image() -> None:
    model = FirstModel()
    price = model.predict(non_squared_image())
    assert np.isclose(price, 2762.0850873685017)


def test_reshape_image() -> None:
    model = FirstModel()
    img = model.reshape_image(non_squared_image())
    assert img.shape[0] == 1
    assert img.shape[1] == img.shape[2]
    assert img.shape[3] == 3
