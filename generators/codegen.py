import os
import torch
from config import *
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenForCausalLM, BitsAndBytesConfig

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 500))


class CodeGenerator:
    _tokenizer = None
    _model = None

    def __init__(self, model_path):
        self.model_path = model_path

    def load_llm(self):
        if CodeGenerator._tokenizer is None or CodeGenerator._model is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            CodeGenerator._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            CodeGenerator._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16, 
                load_in_4bit= True, 
                # quantization_config = quantization_config
                device_map="auto"
            )
        return CodeGenerator._tokenizer, CodeGenerator._model

    def inference(self, prompt, temperature, tokenizer, llm_model):
        """
        Parameters:
            input: encoded prompts
            max_lenght: maximum length of result
            temperature: control randomly
            do_sample: generate sample randomly instead of get best.

        Returns:
            LLM response
        """
        # encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # generate
        sample = llm_model.generate(
            **inputs,
            max_length=MAX_LENGTH + len(inputs[0]),
            temperature=temperature,
            num_beams=1,
            do_sample=True
        )

        # decode prompt
        response = tokenizer.decode(
            sample[0][len(inputs[0]) :],
            truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
            clean_up_tokenization_spaces=True,
        )
        return response