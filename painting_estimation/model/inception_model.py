import numpy as np
import pathlib
import joblib
from painting_estimation.images.preprocessing import ImagePreprocessor, ImgSize
from painting_estimation.inference.inference import ModelServing, ONNXModel


MODELS_DIR = pathlib.Path(__file__).parent.parent / "data/models/1/"


def preprocessor(image: np.ndarray) -> np.ndarray:
    preproc = ImagePreprocessor(
        target_size=ImgSize(width=299, height=299), target_dim_order=(0, 1, 2), target_dtype=np.float32
    )
    image = preproc(image)
    return image


def postprocessor(nn_output: np.ndarray) -> float:
    lgbm = joblib.load(str(MODELS_DIR / "lgb.pkl"))
    price = np.expm1(lgbm.predict(nn_output))[0]
    price = price if price < 10000 else price * 1.5
    return price


FIRST_SERVING: ModelServing = ModelServing(
    model=ONNXModel(str(MODELS_DIR / "incept_v3_1.onnx")),
    preprocessor=preprocessor,
    postprocessor=postprocessor,
)
