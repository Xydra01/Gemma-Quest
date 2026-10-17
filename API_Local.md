# Technical Design Document: Hybrid Inference Architecture (Local vs. Cloud)

## 1. Overview
This document defines the architecture for supporting **dual-mode inference**: running the game with local models via Ollama (for privacy/low latency) or connecting to a cloud-based API (e.g., Google Vertex AI, Together.ai, or Hugging Face Inference Endpoints) for higher quality Gemma 4 variants.

The goal is **seamless switching**. The rest of the application (TUI, State Management, Prompt Engineering) remains unchanged; only the `LLMClient` layer adapts to the backend.

## 2. Design Goals
*   **Abstraction:** Hide the difference between local and remote inference behind a single interface (`InferenceProvider`).
*   **Zero-Cost Switching:** Users can switch modes via a simple config flag or command-line argument without code changes.
*   **Fallback Capability:** If the cloud API fails (rate limit, network), optionally fall back to a local model if available.

## 3. Architecture Diagram

```text
┌─────────────────────────────────────────────────────┐
│                 Application Core                    │
│  (TUI, State Mgmt, Prompt Builder)                  │
└──────────────────────────────┬──────────────────────┘
                               │
                          ┌────▼────┐
                          │         │
                     ┌────▼─────────▼────┐
                     │   LLM Client     │  <-- Abstract Interface
                     │   (Adapter)      │
                     └────┬─────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │ LOCAL MODE │ │ HYBRID     │ │ CLOUD MODE │
    │ (Ollama)   │ │ (Fallback) │ │ (API Key)  │
    │            │ │            │ │            │
    │ • Fast     │ │ • Try Cloud│ │ • High Q&A │
    │ • Private  │ │ • Fallback │ │ • No HW Req│
    └────────────┘ └────────────┘ └────────────┘
```

## 4. Technology Stack Changes

| Component | Local Mode (Ollama) | Cloud Mode (API) |
| :--- | :--- | :--- |
| **Client Library** | `ollama-python` | `requests` / `httpx` + `google-cloud-aiplatform` or generic REST client. |
| **Authentication** | None (Local Socket) | API Key in `.env` file or Env Variable (`GEMMA_API_KEY`). |
| **Latency** | Low (100ms - 2s) | Medium-High (1s - 5s depending on network/model). |
| **Cost** | $0 (Electricity) | Pay-per-token. |

## 5. Implementation: The Abstract Interface

We define a base class `InferenceProvider` that both backends must implement. This allows the main game loop to remain agnostic of the backend.

### 5.1 Base Class Definition

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class InferenceProvider(ABC):
    """Abstract base class for LLM inference backends."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], options: dict = None) -> str:
        """
        Sends a prompt and returns the raw string response.
        
        Args:
            messages: List of message dicts (role, content).
            options: Optional params like temperature, max_tokens.
            
        Returns:
            Raw string response from the model.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the backend is reachable/initialized."""
        pass
```

### 5.2 Local Provider Implementation (Ollama)

```python
import ollama

class LocalProvider(InferenceProvider):
    def __init__(self, model_name: str = "gemma4:2b-q4_0", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        # Initialize client (Ollama python client handles connection)
        
    def is_available(self) -> bool:
        try:
            ollama.list() # Simple ping to check if server is up
            return True
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]], options: dict = None) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **options
            )
            return response["message"]["content"]
        except Exception as e:
            raise ConnectionError(f"Local LLM failed: {e}")
```

### 5.3 Cloud Provider Implementation (Example: Google Vertex AI / Generic API)

*Note: Specific API details vary by provider. This example uses a generic REST approach compatible with many providers.*

```python
import httpx
import os

class CloudProvider(InferenceProvider):
    def __init__(self, api_key: str, endpoint: str, model_id: str = "gemma-4"):
        self.api_key = api_key
        self.endpoint = endpoint # e.g., https://aiplatform.googleapis.com/v1/projects/...
        self.model_id = model_id

    def is_available(self) -> bool:
        return bool(self.api_key) and bool(os.getenv("GEMMA_API_KEY"))

    def chat(self, messages: List[Dict[str, str]], options: dict = None) -> str:
        # Construct payload based on specific API schema (e.g., Vertex AI)
        payload = {
            "messages": messages,
            "model": self.model_id,
            **options
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    # Parse response format (varies by provider)
                    return data.get("content", "") 
                else:
                    raise Exception(f"API Error {resp.status_code}: {resp.text}")
        except Exception as e:
            raise ConnectionError(f"Cloud LLM failed: {e}")
```

## 6. Configuration Management

We use a `config.yaml` or `.env` file to determine the mode.

### 6.1 Config File Structure (`config.yaml`)

```yaml
llm:
  # Mode: 'local', 'cloud', or 'hybrid'
  mode: local

  # Local Settings (Used if mode is local/hybrid)
  local_model: "gemma4:2b-q4_0"
  
  # Cloud Settings (Used if mode is cloud/hybrid)
  cloud_provider: "vertex_ai" # or 'together', 'huggingface'
  cloud_model_id: "gemma-4-instruct"
  api_key_env_var: "GEMMA_API_KEY"
  
  # General
  temperature: 0.8
  max_tokens: 1024
```

### 6.2 Environment Variables (`.env`)

```bash
# Required for Cloud Mode
GEMMA_API_KEY=your_secret_key_here
```

## 7. Hybrid Mode Logic (The "Smart Fallback")

In **Hybrid** mode, the application attempts to use the Cloud first for higher quality, but automatically falls back to Local if:
1.  The API key is missing/invalid.
2.  Network timeout occurs (>5s).
3.  Rate limit error (429) is received.

### 7.1 Provider Factory Pattern

```python
class LLMFactory:
    @staticmethod
    def get_provider(config: dict) -> InferenceProvider:
        mode = config['llm']['mode']
        
        if mode == 'local':
            return LocalProvider(
                model_name=config['llm'].get('local_model', 'gemma4:2b-q4_0')
            )
            
        elif mode == 'cloud':
            api_key = os.getenv(config['llm']['api_key_env_var'])
            if not api_key:
                raise ValueError("API Key not found for cloud mode.")
            return CloudProvider(
                api_key=api_key,
                endpoint=config['llm'].get('endpoint'),
                model_id=config['llm']['cloud_model_id']
            )
            
        elif mode == 'hybrid':
            # Create a wrapper that tries Cloud first, then Local
            cloud_prov = None
            try:
                api_key = os.getenv(config['llm']['api_key_env_var'])
                if api_key:
                    cloud_prov = CloudProvider(api_key, config['llm'].get('endpoint'))
            except Exception:
                pass

            local_prov = LocalProvider(
                model_name=config['llm'].get('local_model')
            )
            
            return HybridWrapper(cloud_prov, local_prov)

class HybridWrapper(InferenceProvider):
    def __init__(self, cloud: Optional[InferenceProvider], local: InferenceProvider):
        self.cloud = cloud
        self.local = local

    def chat(self, messages, options=None):
        if self.cloud and self.cloud.is_available():
            try:
                return self.cloud.chat(messages, options)
            except (ConnectionError, TimeoutException):
                print("[WARN] Cloud failed. Falling back to Local.")

        # Fallback to Local
        return self.local.chat(messages, options)
```

## 8. UI Indicators for Inference Mode

To keep the user informed, the TUI Header should reflect the current inference source.

*   **Local:** `GEMMA DUNGEON [LOCAL: Ollama]`
*   **Cloud:** `GEMMA DUNGEON [CLOUD: Vertex AI]`
*   **Hybrid (Fallback):** `GEMMA DUNGEON [HYBRID -> LOCAL]`

This can be achieved by having the `InferenceProvider` expose a `get_status()` method that returns a short string label.

## 9. Performance & Cost Considerations

| Metric | Local (Ollama) | Cloud (API) |
| :--- | :--- | :--- |
| **Latency** | ~500ms - 2s (CPU/GPU dependent) | ~1s - 3s (Network + Inference) |
| **Consistency** | Varies by hardware load | High (Managed infrastructure) |
| **Cost** | $0 | ~$0.005 - $0.02 per request (Est.) |
| **Privacy** | Data stays on device | Data sent to third party |

**Recommendation for Demo:**
Default to **Local** mode for the demo to ensure zero cost and immediate start-up. Provide a `--use-cloud` CLI flag for users who want to test higher-quality models (like Gemma 4 7B/9B) if they have API credits.

## 10. Security & Best Practices

1.  **API Key Storage:** Never hardcode API keys in source code. Always use environment variables or a local `.env` file that is git-ignored.
2.  **Input Sanitization for Cloud:** When sending data to the cloud, ensure no sensitive local system info (like `/etc/passwd`) is accidentally included in the prompt context.
3.  **Rate Limiting:** If using Hybrid mode, implement a simple "Circuit Breaker" pattern. If the cloud API fails 3 times in a row, stop trying it for the rest of the session to avoid hammering the service or burning through credits unnecessarily.