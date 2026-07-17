"""Ollama chat client: seeded, cached, logged. JSON mode with one corrective retry."""
import json
import time

import requests

from common import CONFIG, cache_get, cache_put, log_call, sha256


def _post_chat(model: str, messages: list, options: dict, json_mode: bool) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if json_mode:
        body["format"] = "json"
    t0 = time.time()
    resp = requests.post(f"{CONFIG['ollama']}/api/chat", json=body,
                         timeout=CONFIG["call_timeout_s"])
    resp.raise_for_status()
    data = resp.json()
    return {
        "content": data["message"]["content"],
        "tokens_in": data.get("prompt_eval_count", 0),
        "tokens_out": data.get("eval_count", 0),
        "seconds": round(time.time() - t0, 2),
    }


def chat(role: str, system: str, user: str, json_mode: bool = True,
         max_tokens: int | None = None) -> dict:
    """One cached call for a configured role. Returns {content, tokens_in, tokens_out, seconds, cached}."""
    spec = CONFIG["roles"][role]
    options = {"temperature": spec["temperature"], "seed": spec["seed"]}
    if max_tokens:
        options["num_predict"] = max_tokens
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    key = sha256({"model": spec["model"], "messages": messages,
                  "options": options, "json": json_mode})
    hit = cache_get(key)
    if hit is not None:
        return {**hit, "cached": True}

    result = _post_chat(spec["model"], messages, options, json_mode)
    log_call({"role": role, "model": spec["model"], "key": key, **result})
    cache_put(key, result)
    return {**result, "cached": False}


def chat_json(role: str, system: str, user: str, required_keys: tuple) -> dict:
    """chat() + parse; one corrective retry if the model emits broken/incomplete JSON."""
    result = chat(role, system, user, json_mode=True)
    parsed = _try_parse(result["content"], required_keys)
    if parsed is not None:
        return {"data": parsed, "meta": result}

    retry_user = (user + "\n\nYour previous output was invalid. Reply with ONLY valid JSON "
                  f"containing the keys: {', '.join(required_keys)}.")
    result = chat(role, system, retry_user, json_mode=True)
    parsed = _try_parse(result["content"], required_keys)
    if parsed is None:
        raise ValueError(f"{role}: unparseable JSON after retry: {result['content'][:200]}")
    return {"data": parsed, "meta": result}


def _try_parse(text: str, required_keys: tuple):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or any(k not in obj for k in required_keys):
        return None
    return obj
