import numpy as np

from painting_estimation.images.preprocessing import image_size
from painting_estimation.inference.inference import ModelServing


def dummy_preprocessor(image: np.ndarray) -> np.ndarray:
    return image


def dummy_postprocessor(nn_output: np.ndarray) -> float:
    img_size = image_size(nn_output)
    return img_size.width * img_size.height / 100


def dummy_model(nn_input: np.ndarray) -> np.ndarray:
    return nn_input


DUMMY_SERVING: ModelServing = ModelServing(
    model=dummy_model,
    preprocessor=dummy_preprocessor,
    postprocessor=dummy_postprocessor,
)
