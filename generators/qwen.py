import transformers
from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeGenerator:
    _tokenizer = None
    _model = None

    def __init__(self, model_path):
        self.model_path = model_path

    def load_llm(self):
        if CodeGenerator._tokenizer is None or CodeGenerator._model is None:
            CodeGenerator._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            CodeGenerator._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto"
            )
        return CodeGenerator._tokenizer, CodeGenerator._model

    def inference(self, prompt, temperature, tokenizer, llm_model):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
