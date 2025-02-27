!pip install transformers torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()
def generate_text(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, top_k=50)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
user_prompt = input("Enter a prompt: ")
generated_paragraph = generate_text(user_prompt, max_length=250)

print("\nGenerated Paragraph based on your prompt:")
print(generated_paragraph)
