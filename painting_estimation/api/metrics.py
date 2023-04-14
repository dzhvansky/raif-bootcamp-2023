import io
import math
import typing

import fastapi
import numpy as np
from PIL import Image
from prometheus_client import Histogram, Summary
from prometheus_fastapi_instrumentator.metrics import Info

from painting_estimation import models


def ml_metrics() -> typing.Callable[[Info], None]:
    ml_prediction_log10 = Histogram(
        "ml_prediction_log10",
        "Distribution of logarithm base 10 of painting price prediction by model.",
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    ml_feature_aspect = Histogram(
        "ml_image_aspect",
        "Distribution of input image aspect.",
        buckets=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
    )
    ml_feature_mean_pixel = Summary("ml_feature_mean_pixel", "Mean value of input image pixels")

    def instrumentation(info: Info) -> None:
        if info.response and info.response.body:
            prediction = models.Predict.parse_raw(info.response.body)
            ml_prediction_log10.observe(math.log10(prediction.price))
            if prediction.aspect:
                ml_feature_aspect.observe(prediction.aspect)
            if prediction.mean_pixel:
                ml_feature_mean_pixel.observe(prediction.mean_pixel)

    return instrumentation


def image_features(file: bytes | io.BytesIO) -> dict:
    image = Image.open(file).convert("RGB")
    return {"aspect": image.width / image.height, "mean_pixel": np.array(image).mean()}
