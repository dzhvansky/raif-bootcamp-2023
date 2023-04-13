import pydantic


class Metrics(pydantic.BaseModel):
    is_alive: bool = True
