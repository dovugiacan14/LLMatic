import os
import torch
from config import *
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenForCausalLM

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 500))


class CodeGenerator:
    _tokenizer = None
    _model = None

    def __init__(self, model_path):
        self.model_path = model_path

    def load_llm(self):
        if CodeGenerator._tokenizer is None or CodeGenerator._model is None:
            CodeGenerator._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            CodeGenerator._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map="auto"
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