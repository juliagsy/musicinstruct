"""Hugging Face Llama client for plan-then-execute inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .plan_executor import build_plan_prompt, parse_plan_text
from .schema import BenchmarkItem, Plan

DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_ENV_LOADED = False


def load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def resolve_hf_token(explicit: str | None = None) -> str | None:
    load_env_file()
    if explicit and explicit.strip():
        return explicit.strip()
    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LlamaPlanClient:
    model_id: str = DEFAULT_LLAMA_MODEL
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    hf_token: str | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._resolved_device: str | None = None
        self._hf_token = resolve_hf_token(self.hf_token)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._resolved_device = resolve_device(self.device)
        dtype = torch.float16 if self._resolved_device == "cuda" else torch.float32
        load_kwargs = {"token": self._hf_token} if self._hf_token else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            **load_kwargs,
        )
        self._model.to(self._resolved_device)
        self._model.eval()

    def propose_plan(self, item: BenchmarkItem) -> tuple[Plan | None, str, dict]:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._resolved_device is not None

        import torch

        prompt = build_plan_prompt(item)
        messages = [
            {
                "role": "system",
                "content": "You convert MIDI edit instructions into strict JSON plans.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self._resolved_device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._resolved_device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **generate_kwargs)

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            plan = parse_plan_text(raw)
        except ValueError:
            return None, raw, {
                "client": "llama",
                "model_id": self.model_id,
                "device": self._resolved_device,
            }
        return plan, raw, {
            "client": "llama",
            "model_id": self.model_id,
            "device": self._resolved_device,
        }
