"""
llm.py — Smart LLM router
  Primary  : Groq API (llama-3.3-70b-versatile / mixtral-8x7b-32768)
  Fallback : TinyLlama 1.1B (local, CPU-safe, ~600 MB)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── lazy-loaded singletons ──────────────────────────────────────────────────
_groq_client = None
_local_model = None
_local_tokenizer = None

GROQ_MODEL = "llama-3.3-70b-versatile"   # swap to "mixtral-8x7b-32768" for longer context
LOCAL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _get_local():
    global _local_model, _local_tokenizer
    if _local_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[LLM] Loading local fallback: {LOCAL_MODEL} ...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print("[LLM] Local model ready.")
    return _local_model, _local_tokenizer


# ── public API ──────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """
    Returns (response_text, model_label).
    Tries Groq first; falls back to local TinyLlama if Groq fails.
    """
    # ── Groq ────────────────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "your_groq_api_key_here":
        try:
            client = _get_groq()
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip(), f"groq/{GROQ_MODEL}"
        except Exception as e:
            print(f"[LLM] Groq failed ({e}), switching to local model …")

    # ── Local fallback ───────────────────────────────────────────────────────
    import torch
    model, tokenizer = _get_local()
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 512),
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )
    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text.strip(), "local/TinyLlama-1.1B"


def stream_generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.2):
    """
    Streams Groq responses. Falls back to local model if Groq fails.
    """
    api_key = os.getenv("GROQ_API_KEY", "")

    if api_key and api_key != "your_groq_api_key_here":
        try:
            client = _get_groq()

            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    yield chunk.choices[0].delta.content

            return

        except Exception as e:
            print(f"[LLM] Groq stream failed ({e}), falling back …")

    # Local fallback
    text, _ = generate(prompt, max_tokens, temperature)
    yield text
