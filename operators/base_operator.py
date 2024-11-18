from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from utils.utils import read_python_file
from generators.codegen import CodeGenerator


@dataclass
class BaseConfig:
    """Base configuration for genetic operators"""

    model_name: str
    num_offspring: int
    temperature: float = 0.7

    @property
    def model_path(self) -> str:
        return f"Salesforce/{self.model_name}"


class BaseGenerator:
    """Base class for mutation and crossover generators"""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.model_generator = CodeGenerator(self.config.model_path)
        self.tokenizer, self.model = self._initialize_model()

    def _initialize_model(self) -> Tuple:
        """Initialize the language model and tokenizer"""
        return self.model_generator.load_llm()

    def _generate_offspring(self, prompt: str) -> List[str]:
        """
        Generate offspring using the provided prompt

        Args:
            prompt: input prompt for generating offspring

        Returns:
            List of generated offspring
        """
        generated_offspring = []
        for _ in range(self.config.num_offspring):
            offspring = self.model_generator.inference(
                prompt=prompt,
                temperature=self.config.temperature,
                tokenizer=self.tokenizer,
                llm_model=self.model,
            )

            generated_offspring.append(offspring)
        return generated_offspring
