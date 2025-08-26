import os, time, json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import yaml
from openai import OpenAI
from rich import print as rprint

load_dotenv()  # loads .env if present

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = load_config()
_client = OpenAI()  # reads OPENAI_API_KEY from env
SAVE_DIR = Path(_cfg.get("project", {}).get("save_dir", "logs"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def call_llm(
    messages,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    retries: int = 3,
    backoff_sec: float = 2.0,
) -> Dict[str, Any]:
    model = model or _cfg["openai"]["model"]
    temperature = temperature if temperature is not None else _cfg["openai"]["temperature"]
    max_tokens = max_tokens or _cfg["openai"]["max_tokens"]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = {
                "text": resp.choices[0].message.content,
                "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage or {}),
                "model": model,
            }
            # simple log
            ts = int(time.time())
            log_path = SAVE_DIR / f"llm_{ts}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            return out
        except Exception as e:
            last_err = e
            rprint(f"[yellow]LLM call failed (attempt {attempt}/{retries}): {e}[/yellow]")
            time.sleep(backoff_sec * attempt)

    raise RuntimeError(f"LLM call repeatedly failed: {last_err}")

