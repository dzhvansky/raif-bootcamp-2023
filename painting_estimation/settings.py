import pydantic


class Settings(pydantic.BaseSettings):
    debug: bool = True


settings: Settings = Settings()
