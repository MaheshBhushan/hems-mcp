"""Bounded OpenAI-compatible HTTP requests; credentials stay in environment variables."""

import json
import math
import os
import time
from urllib.parse import urlsplit

import httpx

EMPTY_USAGE = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def resolve_config(config):
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError("llm_config must be a JSON object")
    if "api_key" in config:
        raise ValueError("Use api_key_env to reference an environment variable")
    base = config.get("base_url", os.getenv("HEMS_LLM_BASE_URL"))
    model = config.get("model", os.getenv("HEMS_LLM_MODEL"))
    if not base or not model:
        return None
    if not isinstance(base, str) or not isinstance(model, str) or not model.strip():
        raise ValueError("base_url and model must be nonempty strings")
    key_env = config.get("api_key_env", "HEMS_LLM_API_KEY")
    if not isinstance(key_env, str) or not key_env or "=" in key_env:
        raise ValueError("api_key_env must name an environment variable")
    temperature = config.get("temperature", 0)
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not 0 <= temperature <= 2 or not math.isfinite(temperature)):
        raise ValueError("temperature must be null or finite between zero and two")
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP(S) URL without credentials or query")
    budget = config.get("max_completion_tokens", 2048)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 16384:
        raise ValueError("max_completion_tokens must be an integer from 1 to 16384")
    return {"base_url": base.rstrip("/"), "model": str(model),
            "api_key_env": key_env,
            "max_completion_tokens": budget, "temperature": temperature}


def request_plan(config, payload):
    """One request, no automatic retry. Return sanitized errors and observed usage."""
    start, cpu = time.perf_counter(), time.process_time()
    result = {"status": "failed", "usage": EMPTY_USAGE.copy(), "provider_compute": None,
              "payload_bytes": len(json.dumps(payload, allow_nan=False).encode())}
    try:
        key = os.getenv(config["api_key_env"])
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": config["model"], "messages": [
            {"role": "system", "content": "Return only a JSON object. Predict nonnegative slot-average kW when requested. Propose charge_kw with one finite nonnegative AC power per future slot. Respect every EV constraint. Do not include markdown."},
            {"role": "user", "content": json.dumps(payload, allow_nan=False)}],
            "max_completion_tokens": config["max_completion_tokens"]}
        if config["temperature"] is not None:
            body["temperature"] = config["temperature"]
        with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=False) as client:  # noqa: SIM117 - stream depends on client
            with client.stream("POST", config["base_url"] + "/chat/completions", json=body, headers=headers) as response:
                if response.status_code >= 400:
                    result["error_category"] = f"http_{response.status_code}"
                    return result
                content = bytearray()
                for chunk in response.iter_bytes():
                    if time.perf_counter() - start > 90:
                        result["error_category"] = "elapsed_budget_exceeded"
                        return result
                    content.extend(chunk)
                    if len(content) > 1_048_576:
                        result["error_category"] = "response_too_large"
                        return result
        envelope = json.loads(content)
        if not isinstance(envelope, dict):
            raise TypeError("invalid envelope")
        usage = envelope.get("usage") or {}
        if not isinstance(usage, dict):
            raise TypeError("invalid usage")
        result["usage"] = {name: value if isinstance(value := usage.get(name), int) and not isinstance(value, bool) and value >= 0 else None for name in EMPTY_USAGE}
        proposal = json.loads(envelope["choices"][0]["message"]["content"])
        if not isinstance(proposal, dict):
            raise TypeError("proposal must be object")
        result.update(status="evaluated", proposal=proposal)
    except httpx.HTTPError:
        result["error_category"] = "transport_error"
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        result["error_category"] = "invalid_response"
    finally:
        result["timing"] = {"wall_seconds": time.perf_counter() - start,
                            "local_cpu_seconds": time.process_time() - cpu}
    return result
