import pydantic


class Predict(pydantic.BaseModel):
    price: float = 2500
