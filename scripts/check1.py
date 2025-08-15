from transformers import AutoModelForCausalLM, AutoTokenizer
m = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype="auto", device_map="auto")
print("loaded:", m)