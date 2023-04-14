import pydantic


class Predict(pydantic.BaseModel):
    price: float = 7000
    aspect: float | None = None
    mean_pixel: float | None = None
