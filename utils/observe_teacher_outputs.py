import torch
from model.model_loader import ModelLoader
from data.data_preparation import TextDataset
import torch.nn.functional as F

def observe_teacher_outputs(
    prompt_file="data/synthetic/prompts_v1.txt",
    model_name="openai-community/gpt2-medium",
    top_k=5
):
    loader = ModelLoader(model_name=model_name)
    dataset = TextDataset(prompt_file)
    
    for idx in range(len(dataset)):
        prompt = dataset[idx]
        outputs = loader.generate_logits([prompt])
        logits = outputs['logits'][0]  # [seq_len, vocab_size]
        last_token_logits = logits[-1]  # Only look at the last token's logits
        
        # Convert logits to probabilities
        probs = F.softmax(last_token_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k)
        
        print(f"\nPrompt {idx+1}: {prompt}")
        print("Top predictions:")
        for i in range(top_k):
            token = loader.tokenizer.decode([topk_indices[i].item()])
            prob = topk_probs[i].item()
            print(f"  {token!r}: {prob:.4f}")

if __name__ == "__main__":
    observe_teacher_outputs()

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")

# prompt = "Explain the concept of knowledge distillation in machine learning."
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

