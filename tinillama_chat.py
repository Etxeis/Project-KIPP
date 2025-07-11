import torch
from transformers import pipeline

# Usa el modelo desde Hugging Face (se descargará la primera vez)
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Puedes cambiar este mensaje para que suene como TARS
messages = [
    {
        "role": "system",
        "content": "Eres TARS, el robot de la película Interstellar. Eres directo, honesto y algo sarcástico, pero siempre profesional.",
    },
    {
        "role": "user",
        "content": "¿Qué es cálculo?",
    }
]

# Aplica el formato para chat (con tokens <|system|> y <|user|>)
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Genera la respuesta
outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# Si quieres extraer solo lo que dijo TARS:
generated = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
print("\n--- Respuesta de TARS ---")
print(generated)
