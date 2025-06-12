import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=0
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the Qwen2 model family."},
]
outputs = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"][-1]['content'])