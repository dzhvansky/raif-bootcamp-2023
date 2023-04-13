import fastapi
from prometheus_fastapi_instrumentator import Instrumentator

from painting_estimation.settings import settings


INSTRUMENTATOR: Instrumentator = Instrumentator()
APP: fastapi.FastAPI = fastapi.FastAPI(debug=settings.debug)
INSTRUMENTATOR.instrument(APP)


@APP.on_event("startup")
async def _startup():
    INSTRUMENTATOR.expose(APP)


@APP.post("/predict")
async def predict(image: fastapi.UploadFile):
    return {"go": "brrr", "image": image.filename}
