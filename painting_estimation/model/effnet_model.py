import pathlib

import joblib
import numpy as np

from painting_estimation.images.preprocessing import ImagePreprocessor, ImgSize
from painting_estimation.inference.inference import ModelServing, ONNXModel


MODELS_DIR = pathlib.Path(__file__).parents[2] / "models/2/"


def preprocessor(image: np.ndarray) -> np.ndarray:
    preproc = ImagePreprocessor(
        target_size=ImgSize(width=299, height=299), target_dim_order=(0, 1, 2), target_dtype=np.float32
    )
    image = preproc(image)
    return image


def postprocessor(nn_output: np.ndarray) -> float:
    lgbm = joblib.load(str(MODELS_DIR / "lgb_new.pkl"))
    price = np.expm1(lgbm.predict(nn_output))[0]
    return price


SECOND_SERVING: ModelServing = ModelServing(
    model=ONNXModel(str(MODELS_DIR / "efn.onnx")),
    preprocessor=preprocessor,
    postprocessor=postprocessor,
)
