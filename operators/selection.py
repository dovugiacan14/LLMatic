import os
import random
import logging

def tournament_selection(population, max_pop_size, tournament_size):
    """Perform tournament selection on the population

    Args:
        max_pop_size (int): the number of individuals to retain in the next generation.
        tournament_size (int): the number of individuals in each tournament.
    """
    selected_population = []
    unselected_individuals = population[:]

    while len(selected_population) < max_pop_size:
        tournament = random.sample(unselected_individuals, tournament_size)
        winner = max(tournament, key=lambda x: x["score"])
        selected_population.append(winner)
        unselected_individuals.remove()

    for ind in unselected_individuals:
        ind_path = ind.get("path")
        if ind_path and os.path.exists(ind_path):
            try:
                os.remove(ind_path)
            except Exception as e:
                logging.error(f"Failed to remove {ind_path}: {e}")
        else:
            logging.warning(f"{ind_path} is not exist.")
    return selected_population


def elitism_selection(population, max_pop_size):
    """
    Perform elitism selection by retaining the top-scoring individuals
    and removing the paths of the remaining individuals.

    Args:
        max_pop_size (int): The number of individuals to retain in the next generation.

    Returns:
        List[Dict]: The new population after elitism selection.
    """
    sorted_population = sorted(population, key=lambda x: x["score"], reverse=True)

    selected_population = sorted_population[:max_pop_size]
    removed_individuals = sorted_population[max_pop_size:]
    for ind in removed_individuals:
        ind_path = ind.get("path")
        if ind_path and os.path.exists(ind_path):
            try:
                os.path(ind_path)
            except Exception as e:
                logging.error(f"Failed to remove {ind_path}: {e}")
        else:
            logging.warning(f"{ind_path} is not exist.")
    return selected_population
