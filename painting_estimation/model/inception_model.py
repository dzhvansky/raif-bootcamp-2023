import pathlib

import joblib
import numpy as np

from painting_estimation.images.preprocessing import ImagePreprocessor
from painting_estimation.images.utils import ImgSize
from painting_estimation.inference.inference import ModelServing, ONNXModel


MODELS_DIR = pathlib.Path(__file__).parent.parent / "data/models/1/"

MODEL = ONNXModel(str(MODELS_DIR / "incept_v3_1.onnx"))
PREPROCESSOR = ImagePreprocessor(
    target_size=ImgSize(width=299, height=299), target_dim_order=(0, 1, 2), target_dtype=np.float32
)


def postprocessor(nn_output: np.ndarray) -> float:
    lgbm = joblib.load(str(MODELS_DIR / "lgb.pkl"))
    price = np.expm1(lgbm.predict(nn_output))[0]
    price = price if price < 10000 else price * 1.5
    return price


FIRST_SERVING: ModelServing = ModelServing(
    model=MODEL,
    preprocessor=PREPROCESSOR,
    postprocessor=postprocessor,
)
