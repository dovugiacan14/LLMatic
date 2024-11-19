from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from operators.base_operator import BaseConfig
from utils.utils import read_python_file
from generators.codegen import CodeGenerator
from .base_operator import BaseGenerator


class Crossover(BaseGenerator):
    """
    Class for crossover generation
    """

    CROSSOVER_PROMPT = """
    Combine the above two neural networks and create a third neural network class that also inherits from nn.Module
    """
    def __init__(self, model_name: str, num_offspring: int, temperature: float= 0.7):
        super().__init__(model_name, num_offspring, temperature)

    def _get_crossover_prompt(self, parent1_code: str, parent2_code: str) -> str: 
        """
        Construct the prompt for crossover 

        Args: 
            parent1_code: code the first parent 
            parent2_code: code of the second parent 
        
        Returns:
            crossover-specific prompt 
        
        """
        return f"{parent1_code}\n{parent2_code}\n{self.CROSSOVER_PROMPT}"
    
    def _act(self, parent1_path: Path, parent2_path: Path) -> List[str]:
        """
        Perform crossover on two code files 

        Args: 
            parent1_path: Path to the first parent's code. 
            parent2_path: Path to the second parent's code. 
        
        Returns: 
            List of offspring generated by crossover.
        """
        parent1_code = read_python_file(parent1_path)
        parent2_code = read_python_file(parent2_path)
        prompt = self._get_crossover_prompt(parent1_code, parent2_code)
        return self._generate_offspring(prompt)