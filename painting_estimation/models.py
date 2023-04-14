import pydantic


class Predict(pydantic.BaseModel):
    price: float = 2500
    aspect: float | None = None
    mean_pixel: float | None = None
