"""Secure key storage helpers for the local QA app.

- Uses the operating system keyring so provider API keys do not need to live in
  tracked files.
- Exposes small read/write helpers used by configuration loading and one-time
  setup flows.
- Does not know anything about corpus loading, retrieval, or UI behavior.
"""

from __future__ import annotations

import keyring
from keyring.errors import KeyringError


class SecretStoreError(RuntimeError):
    """Raised when a required secret is missing or the keyring backend fails."""


def load_secret(service: str, username: str) -> str:
    """Load one secret from the operating system keyring."""

    try:
        secret = keyring.get_password(service, username)
    except KeyringError as error:
        raise SecretStoreError(f"Unable to read secret from keyring: {error}") from error
    if secret is None or not secret.strip():
        raise SecretStoreError(
            f"No stored secret found in keyring for service='{service}' username='{username}'"
        )
    return secret.strip()


def save_secret(service: str, username: str, secret: str) -> None:
    """Persist one secret into the operating system keyring."""

    normalized_secret = secret.strip()
    if not normalized_secret:
        raise SecretStoreError("Cannot store an empty secret in the keyring")
    try:
        keyring.set_password(service, username, normalized_secret)
    except KeyringError as error:
        raise SecretStoreError(f"Unable to store secret in keyring: {error}") from error


__all__ = ["SecretStoreError", "load_secret", "save_secret"]
