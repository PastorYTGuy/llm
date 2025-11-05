import os
import time
from typing import List

import tiktoken
import torch.cuda
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", )

# Konfiguracja Gemini
genai.configure(api_key=API_KEY)

# Ładowanie lokalnego modelu (opcjonalnie)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Klient OpenAI / Groq
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=BASE_URL
)

genai.configure(api_key=API_KEY)



def chat_once_openai(prompt: str,
                     system: str = "You are a helpful assistant",
                     temperature: float = 0.8,
                     top_p: float = 0.7,
                     max_output_tokens: int = 512):
    t0 = time.time()
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens
    )
    dt = time.time() - t0
    choice = response.choices[0]
    out = {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
        "latency_s": round(dt, 3),
        "usage": getattr(response, "usage", None) and response.usage.model_dump()
    }
    return out


def approx_tokens_openai(texts: List[str], model_name: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return sum(len(enc.encode(t)) for t in texts)
    except Exception:
        return sum(max(1, len(t) // 4) for t in texts)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int,
                      price_in: float = 0.000005, price_out: float = 0.000015):
    return prompt_tokens * price_in + completion_tokens * price_out


def chat_once_gemini(prompt: str,
                     system: str = "You are a helpful assistant",
                     temperature: float = 0.8,
                     top_p: float = 0.7,
                     top_k: int = 40,
                     max_output_tokens: int = 512):
    """
    Wywołuje model Gemini (Google Generative AI)
    """
    t0 = time.time()

    model = genai.GenerativeModel(MODEL_NAME)
    config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens
    )

    response = model.generate_content(
        [system, prompt],
        generation_config=config
    )

    dt = time.time() - t0
    out = {
        "text": response.text,
        "latency_s": round(dt, 3),
        "candidates": getattr(response, "candidates", None)
    }
    return out


if __name__ == "__main__":
    user_input = input("Podaj prompt: ")
    output = chat_once_openai(user_input, system="You are a helpful assistant")
    output_txt = output["text"]
    ptoks = approx_tokens_openai([user_input])
    otoks = approx_tokens_openai([output_txt])
    print(f"Prompt tokens: {ptoks}, Output tokens: {otoks}")
    print(f"Szacowany koszt: ${estimate_cost_usd(ptoks, otoks):.6f}")
    print("\nOdpowiedź modelu:\n", output_txt)
    print("\nUżycie:", output.get("usage"))