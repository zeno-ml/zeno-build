"""A module for global variables regarding models."""

from __future__ import annotations

import cohere

cohere_client: cohere.Client | None = None
