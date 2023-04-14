import pathlib
import random

from locust import HttpUser, task

from painting_estimation.settings import settings


IMAGE_PATHS = list((pathlib.Path(__file__).parents[1] / "downloads" / "artsynet" / "images").glob("*.jpg"))


class PredictAPIUser(HttpUser):
    host = settings.ml_api.replace("/predict", "")

    @task
    def fetch_from_predict_api(self):
        one_image_path = random.choice(IMAGE_PATHS)
        self.client.post(settings.ml_api, files={"file": one_image_path.read_bytes()})
