import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ChatModel:
    def __init__(self, model_id="google/gemma-2b", device="cuda:0"):
        # Load the tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Use quantization for memory optimization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        # Load the model from Hugging Face hub
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # Automatically distribute model across available devices
            quantization_config=quantization_config,
        )

        self.model.eval()
        self.device = device

    def inference(self, question: str, context: str = None, max_new_tokens: int = 1200):
        # Define the prompt based on the context or just the question
        if not context:
            prompt = (
                f"You are Bot assistant powered by the Gemma language model. "
                f"Provide a detailed and accurate answer to the following question.\n\n"
                f"Question: {question}\n"
            )
        else:
            prompt = (
                f"You are Bot assistant powered by the Gemma language model. "
                f"Provide a detailed answer using the provided context.\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n\n"
            )

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=True)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

