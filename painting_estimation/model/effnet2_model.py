import pathlib

import cv2
import joblib
import numpy as np

from painting_estimation.images.preprocessing import ImagePreprocessor, ImgSize
from painting_estimation.inference.inference import ModelServing, ONNXModel


MODELS_DIR = pathlib.Path(__file__).parents[2] / "models/3/"

EFF_NET_ONNX_FEATURE_EXTRACTOR = ONNXModel(str(MODELS_DIR / "eff_net_b3.onnx.onnx"))
LIGHT_GBM_REGR = joblib.load(str(MODELS_DIR / "lgbm.pkl"))


EFF_NET_PREPROCESSOR = ImagePreprocessor(
    target_size=ImgSize(width=300, height=300),
    target_dim_order=(2, 0, 1),
    target_dtype=np.float32,
    interpolation=cv2.INTER_LINEAR,
    to_bgr=False,
    extra_batch_dim=0,
    normalize=True,
    means=(0.485, 0.456, 0.406),
    stds=(0.229, 0.224, 0.225),
    initial_size_before_crop=ImgSize(width=320, height=320),
)


def postprocessor(nn_output: np.ndarray) -> float:
    lgbm = LIGHT_GBM_REGR
    price = np.expm1(lgbm.predict(nn_output))[0]
    return price


THIRD_SERVING: ModelServing = ModelServing(
    model=EFF_NET_ONNX_FEATURE_EXTRACTOR,
    preprocessor=EFF_NET_PREPROCESSOR,
    postprocessor=postprocessor,
)
