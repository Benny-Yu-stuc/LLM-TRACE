"""Optional external Chief-Engineer client.

The API key and URL are intentionally blank in the default configuration.
When either value is missing, the caller should use the local router.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any


class ChiefEngineerApiClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.api_key = config.get("api_key", "")
        self.api_url = config.get("api_url", "")
        self.model = config.get("model", "")
        self.temperature = config.get("temperature", 0.1)
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 4096)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.api_url)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Chief-Engineer API is not configured.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = _extract_content(data)
        if isinstance(content, dict):
            return content
        return json.loads(str(content))


def _extract_content(data: dict[str, Any]) -> Any:
    if "choices" in data:
        choice = data["choices"][0]
        message = choice.get("message", {})
        return message.get("content", choice.get("text", "{}"))
    if "output" in data:
        return data["output"]
    return data
