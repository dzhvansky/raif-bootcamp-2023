import os
import typing

import pydantic


class Settings(pydantic.BaseSettings):
    debug: bool = True
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "===WRONG_TELEGRAM_TOKEN===")
    flyio_host: str = "https://velvet-wolves-art-expert-api.fly.dev"
    flyio_port: typing.Optional[int] = None


settings: Settings = Settings()
