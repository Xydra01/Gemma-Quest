"""Factory pattern for instantiating LLM inference providers."""

import logging
from typing import Optional
from src.config import GameConfig, load_config
from .base import InferenceAuthError, InferenceProvider
from .cloud import CloudProvider
from .hybrid import HybridWrapper
from .local import LocalProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Instantiates configured InferenceProvider based on GameConfig settings."""

    @staticmethod
    def get_provider(
        config: Optional[GameConfig] = None,
        mode_override: Optional[str] = None,
    ) -> InferenceProvider:
        """Constructs and returns an InferenceProvider instance.

        Args:
            config: Optional GameConfig. If None, loads from config.yaml / .env.
            mode_override: Optional mode ('local', 'cloud', 'hybrid') overriding config.

        Returns:
            Configured InferenceProvider (LocalProvider, CloudProvider, or HybridWrapper).

        Raises:
            InferenceAuthError: If Cloud mode is requested without an API key.
            ValueError: If an unrecognized mode is specified.
        """
        cfg = config or load_config()
        mode = (mode_override or cfg.llm.mode).lower().strip()

        # Local Mode
        if mode == "local":
            return LocalProvider(
                model_name=cfg.llm.local_model,
                base_url=cfg.llm.ollama_base_url,
                timeout_seconds=cfg.llm.timeout_seconds,
            )

        # Cloud Mode
        elif mode == "cloud":
            api_key = cfg.llm.get_api_key()
            if not api_key:
                raise InferenceAuthError(
                    f"Cloud inference requested, but environment variable '{cfg.llm.api_key_env_var}' "
                    "is not set. Please set your API key in .env or switch to local mode."
                )

            return CloudProvider(
                api_key=api_key,
                endpoint=cfg.llm.cloud_endpoint,
                model_id=cfg.llm.cloud_model_id,
                provider_name=cfg.llm.cloud_provider,
                timeout_seconds=cfg.llm.timeout_seconds,
            )

        # Hybrid Mode
        elif mode == "hybrid":
            local_prov = LocalProvider(
                model_name=cfg.llm.local_model,
                base_url=cfg.llm.ollama_base_url,
                timeout_seconds=cfg.llm.timeout_seconds,
            )

            cloud_prov = None
            api_key = cfg.llm.get_api_key()
            if api_key:
                cloud_prov = CloudProvider(
                    api_key=api_key,
                    endpoint=cfg.llm.cloud_endpoint,
                    model_id=cfg.llm.cloud_model_id,
                    provider_name=cfg.llm.cloud_provider,
                    timeout_seconds=cfg.llm.timeout_seconds,
                )
            else:
                logger.warning(
                    "Hybrid mode configured, but cloud API key or endpoint is missing. Operating locally."
                )

            return HybridWrapper(
                local=local_prov,
                cloud=cloud_prov,
                fallback_timeout=cfg.resilience.hybrid_fallback_timeout,
                circuit_breaker_threshold=cfg.resilience.circuit_breaker_threshold,
            )

        else:
            raise ValueError(f"Unsupported inference mode: '{mode}'. Must be 'local', 'cloud', or 'hybrid'.")
