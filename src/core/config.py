from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Market data APIs ---
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""

    # --- Trading simulation ---
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.001
    commission_per_trade: float = 1.0
    commission_pct: float = 0.0005
    max_position_pct: float = 0.20

    # --- Application ---
    log_level: str = "INFO"
    data_dir: Path = Path("src/data")
    max_agent_iterations: int = 3

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def traces_db_path(self) -> Path:
        return self.data_dir / "traces.db"

    @property
    def faiss_index_dir(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def portfolio_path(self) -> Path:
        return self.data_dir / "portfolio.json"


settings = Settings()
