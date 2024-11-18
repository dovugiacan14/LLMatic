from pathlib import Path
from typing import List, Tuple
from utils.utils import read_python_file
from generators.codegen import CodeGenerator
from .base_operator import BaseGenerator

class Mutation(BaseGenerator):
    # """Class for mutation generation."""
    # def __init__(self, model_name: str, num_offspring: int, temperature: float= 0.7):
    #     super().__init__(model_name, num_offspring, temperature)

    def _act(self, code_path: Path, prompt: str) -> List[str]:
        """
        Perform mutation on the given code

        Args: 
            code_path: Path to the code file to mutate 
            prompt: Mutation-specific prompt 

        Returns: 
            List of mutated offspring 
        """
        code_string = read_python_file(code_path)
        full_prompt = f"{code_string}\n{[prompt]}"
        return self._generate_offspring(full_prompt)