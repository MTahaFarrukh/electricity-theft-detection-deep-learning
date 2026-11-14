"""Settings read from the environment / .env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_env: str = "development"
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'backend' / 'etd.db').as_posix()}"
    cors_origins: str = "http://localhost:5173"
    model_metadata_path: Path = PROJECT_ROOT / "artifacts" / "metadata" / "model.json"
    results_dir: Path = PROJECT_ROOT / "reports" / "results"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
