import fastapi
from painting_estimation.settings import settings
from painting_estimation import models

APP: fastapi.FastAPI = fastapi.FastAPI(debug=settings.debug)


@APP.get("/metrics", response_model=models.Metrics)
async def metrics():
    return models.Metrics(is_alive=True)


@APP.post("/predict")
async def predict(image: fastapi.UploadFile):
    return {"go": "brrr", "image": image.filename}
