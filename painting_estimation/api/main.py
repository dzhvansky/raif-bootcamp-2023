import logging

import fastapi
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import metrics as fastapi_metrics

from painting_estimation import models
from painting_estimation.api import metrics
from painting_estimation.inference.serving import predict_painting_price
from painting_estimation.settings import settings


LOGGER: logging.Logger = logging.getLogger(__name__)
APP: fastapi.FastAPI = fastapi.FastAPI(debug=settings.debug)
INSTRUMENTATOR: Instrumentator = Instrumentator(body_handlers=["/predict"])
INSTRUMENTATOR.add(metrics.ml_metrics(), fastapi_metrics.default())
INSTRUMENTATOR.instrument(APP)


@APP.on_event("startup")
async def _():
    INSTRUMENTATOR.expose(APP)


@APP.post("/predict", response_model=models.Predict)
async def predict(file: fastapi.UploadFile):
    LOGGER.info(f"Got image `{file.filename}` with type `{file.content_type}`")
    try:
        price: float = predict_painting_price(byte_io=file.file)  # type: ignore
    except Exception:
        LOGGER.error("Some error happened during prediction with model!", exc_info=True)
        return models.Predict()
    return models.Predict(price=price, **metrics.image_features(file.file))  # type: ignore
