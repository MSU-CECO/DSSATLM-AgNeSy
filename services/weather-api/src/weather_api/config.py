from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    weather_parquet_path: str
    weather_api_key: str
    host: str = "0.0.0.0"
    port: int = 8001


settings = Settings()
