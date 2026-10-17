"""Unit and verification tests for inference providers, hybrid routing, and factory."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.config import GameConfig
from src.engine import (
    CloudProvider,
    HybridWrapper,
    InferenceAuthError,
    InferenceConnectionError,
    InferenceError,
    InferenceProvider,
    InferenceRateLimitError,
    InferenceTimeoutError,
    LLMFactory,
    LocalProvider,
)


class MockProvider(InferenceProvider):
    """Simple mock provider for unit testing."""

    def __init__(self, name: str = "mock", available: bool = True, canned_response: str = "Mock output"):
        self.name = name
        self.available = available
        self.canned_response = canned_response
        self.call_count = 0
        self.last_options = None

    def is_available(self) -> bool:
        return self.available

    def chat(self, messages, options=None):
        self.call_count += 1
        self.last_options = options
        return self.canned_response

    def get_status(self) -> str:
        return f"[MOCK: {self.name}]"


# --- Local Provider Tests ---

def test_local_provider_options_mapping():
    """Verify that generic sampling options are properly translated for Ollama."""
    provider = LocalProvider(model_name="gemma4:2b-q4_0")
    provider.client = MagicMock()
    provider.client.chat.return_value = {"message": {"content": "The tavern is lively."}}

    messages = [{"role": "user", "content": "Look around"}]
    options = {"temperature": 0.7, "top_p": 0.85, "top_k": 40, "max_tokens": 512}

    response = provider.chat(messages, options)

    assert response == "The tavern is lively."
    provider.client.chat.assert_called_once()
    called_args = provider.client.chat.call_args[1]
    assert called_args["model"] == "gemma4:2b-q4_0"
    assert called_args["options"] == {
        "temperature": 0.7,
        "top_p": 0.85,
        "top_k": 40,
        "num_predict": 512,
    }


def test_local_provider_connection_error():
    """Verify mapping of connection failure to InferenceConnectionError."""
    provider = LocalProvider(base_url="http://invalid-host:11434")
    provider.client = MagicMock()
    provider.client.chat.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(InferenceConnectionError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_local_provider_timeout_error():
    """Verify mapping of timeout failure to InferenceTimeoutError."""
    provider = LocalProvider()
    provider.client = MagicMock()
    provider.client.chat.side_effect = httpx.TimeoutException("Read timed out")

    with pytest.raises(InferenceTimeoutError):
        provider.chat([{"role": "user", "content": "hi"}])


def test_local_provider_availability():
    """Verify is_available() probes Ollama client."""
    provider = LocalProvider()

    with patch("ollama.Client.list", return_value={"models": []}):
        assert provider.is_available() is True

    with patch("ollama.Client.list", side_effect=Exception("Server offline")):
        assert provider.is_available() is False


# --- Cloud Provider Tests ---

def test_cloud_provider_success():
    """Verify successful cloud chat completion."""
    provider = CloudProvider(
        api_key="test-api-key",
        endpoint="https://api.example.com/v1/chat/completions",
        model_id="gemma-4-instruct",
        provider_name="vertex_ai",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The shadows deepen."}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = provider.chat([{"role": "user", "content": "Hello"}])
        assert result == "The shadows deepen."

        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-api-key"

    assert provider.get_status() == "[CLOUD: vertex_ai]"


def test_cloud_provider_google_ai_protocol():
    """Verify Google AI Studio generateContent payload format and header."""
    provider = CloudProvider(
        api_key="test-google-key",
        endpoint="",  # Default Google AI Studio endpoint
        model_id="gemma-4-26b-a4b-it",
        provider_name="google_ai",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "A cold draft fills the hallway."}]}}
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        messages = [
            {"role": "system", "content": "You are a DM."},
            {"role": "user", "content": "I open the door."},
        ]
        result = provider.chat(messages)
        assert result == "A cold draft fills the hallway."

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["x-goog-api-key"] == "test-google-key"
        assert "systemInstruction" in call_kwargs["json"]
        assert "contents" in call_kwargs["json"]


def test_cloud_provider_google_ai_thinking_parts():
    """Verify that Google AI extracts non-thought completion parts over thought parts."""
    provider = CloudProvider(
        api_key="test-key",
        endpoint="",
        model_id="gemma-4-26b-a4b-it",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "* Internal reasoning monologue *"},
                        {"text": '{"narration": "You stand before the ancient gate."}'},
                    ]
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        result = provider.chat([{"role": "user", "content": "look"}])
        assert result == '{"narration": "You stand before the ancient gate."}'
        assert "* Internal reasoning" not in result


def test_cloud_provider_rejects_empty_response():
    """Verify that CloudProvider raises InferenceError on empty response."""
    provider = CloudProvider(api_key="test-key", endpoint="https://api.example.com")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": "   "}}]}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(InferenceError, match="empty or whitespace response"):
            provider.chat([{"role": "user", "content": "hi"}])


def test_cloud_provider_http_errors():
    """Verify translation of HTTP error statuses."""
    provider = CloudProvider(api_key="key", endpoint="https://example.com")

    # 401 Unauthorized
    resp_401 = MagicMock(status_code=401, text="Unauthorized")
    with patch("httpx.Client.post", return_value=resp_401):
        with pytest.raises(InferenceAuthError):
            provider.chat([{"role": "user", "content": "hi"}])

    # 429 Rate Limit
    resp_429 = MagicMock(status_code=429, text="Too Many Requests")
    with patch("httpx.Client.post", return_value=resp_429):
        with pytest.raises(InferenceRateLimitError):
            provider.chat([{"role": "user", "content": "hi"}])

    # 500 Server Error
    resp_500 = MagicMock(status_code=500, text="Internal Server Error")
    with patch("httpx.Client.post", return_value=resp_500):
        with pytest.raises(InferenceConnectionError):
            provider.chat([{"role": "user", "content": "hi"}])


def test_cloud_provider_timeout():
    """Verify cloud timeout translation."""
    provider = CloudProvider(api_key="key", endpoint="https://example.com")
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(InferenceTimeoutError):
            provider.chat([{"role": "user", "content": "hi"}])


# --- Hybrid Wrapper Tests ---

def test_hybrid_provider_routes_to_cloud():
    """Verify that hybrid mode normally executes Cloud and preserves status."""
    local = MockProvider("local", canned_response="Local text")
    cloud = MockProvider("cloud", canned_response="Cloud text")

    hybrid = HybridWrapper(local=local, cloud=cloud, fallback_timeout=5.0)
    assert hybrid.get_status() == "[HYBRID: Cloud Active]"

    result = hybrid.chat([{"role": "user", "content": "Hi"}])
    assert result == "Cloud text"
    assert cloud.call_count == 1
    assert local.call_count == 0
    assert hybrid.consecutive_failures == 0


def test_hybrid_provider_fallback_on_failure():
    """Verify that hybrid mode immediately falls back to local on cloud failure."""
    local = MockProvider("local", canned_response="Local fallback text")
    cloud = MagicMock(spec=InferenceProvider)
    cloud.is_available.return_value = True
    cloud.chat.side_effect = InferenceTimeoutError("Cloud timed out")

    hybrid = HybridWrapper(local=local, cloud=cloud, fallback_timeout=5.0, circuit_breaker_threshold=3)

    result = hybrid.chat([{"role": "user", "content": "Hi"}])

    assert result == "Local fallback text"
    assert cloud.chat.call_count == 1
    assert local.call_count == 1
    assert hybrid.consecutive_failures == 1
    assert hybrid.circuit_open is False
    assert hybrid.get_status() == "[HYBRID -> LOCAL (Fallback)]"


def test_hybrid_provider_circuit_breaker_trips():
    """Verify that circuit breaker opens after 3 consecutive failures and locks to Local."""
    local = MockProvider("local", canned_response="Local text")
    cloud = MagicMock(spec=InferenceProvider)
    cloud.is_available.return_value = True
    cloud.chat.side_effect = InferenceRateLimitError("Rate limit 429")

    hybrid = HybridWrapper(local=local, cloud=cloud, fallback_timeout=5.0, circuit_breaker_threshold=3)

    # 1st failure
    hybrid.chat([{"role": "user", "content": "1"}])
    assert hybrid.consecutive_failures == 1
    assert not hybrid.circuit_open

    # 2nd failure
    hybrid.chat([{"role": "user", "content": "2"}])
    assert hybrid.consecutive_failures == 2
    assert not hybrid.circuit_open

    # 3rd failure: Trips circuit breaker
    hybrid.chat([{"role": "user", "content": "3"}])
    assert hybrid.consecutive_failures == 3
    assert hybrid.circuit_open is True
    assert hybrid.get_status() == "[HYBRID: Circuit Open (Local)]"

    # 4th call: Must route directly to local WITHOUT touching cloud
    cloud.chat.reset_mock()
    result = hybrid.chat([{"role": "user", "content": "4"}])
    assert result == "Local text"
    cloud.chat.assert_not_called()

    # Reset circuit breaker
    hybrid.reset_circuit_breaker()
    assert hybrid.circuit_open is False
    assert hybrid.consecutive_failures == 0


# --- Factory Tests ---

def test_llm_factory_modes(monkeypatch):
    """Verify factory construction across local, cloud, and hybrid configurations."""
    cfg = GameConfig()

    # Local mode
    cfg.llm.mode = "local"
    prov_local = LLMFactory.get_provider(cfg)
    assert isinstance(prov_local, LocalProvider)

    # Cloud mode with key
    cfg.llm.mode = "cloud"
    cfg.llm.cloud_endpoint = "https://api.example.com"
    monkeypatch.setenv("GEMMA_API_KEY", "secret-key")
    prov_cloud = LLMFactory.get_provider(cfg)
    assert isinstance(prov_cloud, CloudProvider)

    # Cloud mode missing key raises InferenceAuthError
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    with pytest.raises(InferenceAuthError):
        LLMFactory.get_provider(cfg)

    # Hybrid mode
    cfg.llm.mode = "hybrid"
    prov_hybrid = LLMFactory.get_provider(cfg)
    assert isinstance(prov_hybrid, HybridWrapper)


def test_cloud_provider_vertex_ai_mode():
    """Verify that CloudProvider routes to Vertex AI client when use_vertexai is True."""
    provider = CloudProvider(
        use_vertexai=True,
        project_id="test-project",
        location="us-central1",
        model_id="gemini-2.5-flash",
    )
    assert provider.is_available() is True
    assert "Vertex AI" in provider.get_status()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"narration": "You stand in a sunlit glade."}'
    mock_client.models.generate_content.return_value = mock_response
    provider._genai_client = mock_client

    result = provider.chat([{"role": "user", "content": "Look around"}])
    assert result == '{"narration": "You stand in a sunlit glade."}'
    mock_client.models.generate_content.assert_called_once()


def test_llm_factory_vertex_ai_mode(monkeypatch):
    """Verify that LLMFactory instantiates Vertex AI CloudProvider when project is configured."""
    cfg = GameConfig()
    cfg.llm.mode = "cloud"
    cfg.visualizer.backend = "vertex_ai"
    cfg.visualizer.project_id = "test-project-123"

    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    provider = LLMFactory.get_provider(cfg)
    assert isinstance(provider, CloudProvider)
    assert provider.use_vertexai is True
    assert provider.project_id == "test-project-123"

