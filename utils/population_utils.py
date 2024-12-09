import random
from typing import List, Dict


def ranking_individuals_in_pop(pop: List[Dict]):
    return sorted(pop, key=lambda x: x["score"], reverse=True)


def select_individuals_mutation(pop: List[Dict]):
    n_pop = len(pop)
    if n_pop > 2:
        weights = [0.5, 0.2] + [0.5 / (n_pop - 2)] * (n_pop - 2)
    else:
        weights = [0, 5, 0.2][:n_pop]

    random_inviduals = random.choices(pop, weights=weights, k=1)[0]
    return random_inviduals