"""
Centralized application configuration.

This is the ONLY module that touches environment variables directly.
Every other module imports get_settings() and works with typed Python objects.

Pattern: Pydantic Settings (https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All PRmate configuration loaded from environment variables or .env file.

    Pydantic v2 validates types at load time — if a required setting is missing
    or has the wrong type, the app fails fast at startup with a clear error,
    NOT four hours later when a webhook arrives.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Pydantic Settings configuration
    # ─────────────────────────────────────────────────────────────────────────
    # - env_file: load from .env in the project root during local development
    # - env_file_encoding: explicit, avoids Windows/Linux UTF-8 surprises
    # - case_sensitive=False: GITHUB_APP_ID and github_app_id both work
    # - extra="ignore": don't crash on unrelated env vars (e.g. PATH, HOME)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # GitHub App credentials
    # ─────────────────────────────────────────────────────────────────────────
    # App ID is an integer — Pydantic will convert "3797115" → 3797115 automatically.
    # If someone sets it to "abc", we crash at startup with a clear TypeError.
    github_app_id: int = Field(
        ...,  # ... means "required, no default"
        description="GitHub App ID, visible on the App's settings page",
    )

    # Client ID is a string (e.g. "Iv23li4RPpaLZepmkDqL").
    # Not strictly secret (it's visible in OAuth redirects) but treat as opaque.
    github_client_id: str = Field(
        ...,
        description="GitHub App Client ID, used for OAuth user identification (V6+)",
    )

    # The PRIVATE KEY is the most sensitive credential — it lets PRmate
    # mint JWTs to impersonate the GitHub App. Stored base64-encoded because:
    #   1. .env files don't handle multi-line strings well
    #   2. Render env vars definitely don't handle multi-line strings
    # We base64-decode it in github/auth.py when generating JWTs.
    #
    # SecretStr wraps the value so it never appears in logs or repr().
    github_private_key_b64: SecretStr = Field(
        ...,
        description="Base64-encoded RSA private key for the GitHub App (from .pem file)",
    )

    # The webhook secret is used to verify that incoming webhooks
    # actually came from GitHub (via HMAC-SHA256 signature check).
    # If this is wrong, every webhook will be rejected — silently catastrophic.
    github_webhook_secret: SecretStr = Field(
        ...,
        description="Shared secret for verifying webhook signatures (X-Hub-Signature-256)",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Application runtime settings
    # ─────────────────────────────────────────────────────────────────────────
    # The environment name controls behavior like log verbosity and which
    # checkpointer to use. We'll use this in V6 to swap SQLite → Postgres.
    environment: str = Field(
        default="development",
        description="One of: development, production",
    )

    # SQLite database file path for the LangGraph checkpointer (V1-V5).
    # In V6 we swap this for a PostgresSaver using DATABASE_URL.
    # Using Path makes filesystem operations explicit and cross-platform.
    sqlite_db_path: Path = Field(
        default=Path("data/prmate.db"),
        description="Path to SQLite database file for LangGraph checkpointer",
    )
    test_installation_id: Optional[int] = Field(
        default=None,
        description="GitHub App installation ID for smoke tests against a real PR",
    )

    test_repo_full_name: Optional[str] = Field(
        default=None,
        description="Repository full_name (owner/repo) for smoke tests",
    )

    test_pr_number: Optional[int] = Field(
        default=None,
        description="PR number on test_repo_full_name to use in smoke tests",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ─────────────────────────────────────────────────────────────────────────
    # Properties let other modules write `settings.is_production` instead of
    # `settings.environment == "production"` — cleaner and less error-prone.
    @property
    def is_production(self) -> bool:
        """True when running on Render (or any non-development environment)."""
        return self.environment.lower() == "production"

    @property
    def smoke_test_ready(self) -> bool:
        """True iff all three smoke test fields are populated in .env."""
        return all(
            x is not None
            for x in (
                self.test_installation_id,
                self.test_repo_full_name,
                self.test_pr_number,
            )
        )

    @property
    def is_development(self) -> bool:
        """True when running locally."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Why a function instead of a module-level `settings = Settings()`?
      1. Lazy loading — settings are only parsed when first needed,
         which matters for tests that may want to override env vars.
      2. Caching via lru_cache — every call after the first is free.
      3. Easy to mock in tests by calling get_settings.cache_clear()
         and patching environment variables.

    Usage in other modules:
        from app.config import get_settings
        settings = get_settings()
        print(settings.github_app_id)
    """
    return Settings()


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — run `python -m app.config` to verify your .env is correct
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This block only runs when the file is executed directly,
    # not when imported by other modules.
    settings = get_settings()

    # Print non-secret values to confirm loading works.
    # SecretStr fields will print as '**********' — that's the point.
    print("=" * 60)
    print("PRmate configuration loaded successfully ✓")
    print("=" * 60)
    print(f"  App ID:           {settings.github_app_id}")
    print(f"  Client ID:        {settings.github_client_id}")
    print(f"  Private key:      {settings.github_private_key_b64}")
    print(f"  Webhook secret:   {settings.github_webhook_secret}")
    print(f"  Environment:      {settings.environment}")
    print(f"  SQLite DB path:   {settings.sqlite_db_path}")
    print(f"  Is production?    {settings.is_production}")
    print("=" * 60)
