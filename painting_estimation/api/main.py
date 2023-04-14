import logging

import fastapi
from prometheus_fastapi_instrumentator import Instrumentator

from painting_estimation import models
from painting_estimation.inference.serving import predict_painting_price
from painting_estimation.settings import settings


LOGGER: logging.Logger = logging.getLogger(__name__)
INSTRUMENTATOR: Instrumentator = Instrumentator()
APP: fastapi.FastAPI = fastapi.FastAPI(debug=settings.debug)
INSTRUMENTATOR.instrument(APP)


@APP.on_event("startup")
async def _():
    INSTRUMENTATOR.expose(APP)


@APP.post("/predict", response_model=models.Predict)
async def predict(file: fastapi.UploadFile):
    LOGGER.info(f"Got image `{file.filename}` with type `{file.content_type}`")
    price: float = predict_painting_price(byte_io=file.file)  # type: ignore
    return models.Predict(price=price)
