import pydantic


class Settings(pydantic.BaseSettings):
    debug: bool = True
    telegram_token: str = "===WRONG_TELEGRAM_TOKEN==="
    ml_api: str = "https://velvet-wolves-art-expert-api.fly.dev/predict"


settings: Settings = Settings()
