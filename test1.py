import os
import random
import logging 

networks_metadata = [
    {
        'net': "",
        'prompt': '"""Add normalization layer to improve the above network, can train on Cifar10. Only output the class definition with its methods."""',
        'path': './database\\network_7.py',
        'score': 315.63017094563065
    },
    {
        'net': "",
        'prompt': '"""Delete a layer to improve the above network, can train on Cifar10. Only output the class definition with its methods."""',
        'path': './database\\network_11.py',
        'score': 217.33834587791316
    },
    {
        'net': "",
        'prompt': '"""improve the above network by increasing the size drastically, can train on Cifar10. Only output the class definition with its methods."""',
        'path' : './database\\network_12.py',
        'score': 2489.927861310226
    },
    {
        'net': "",
        'prompt': '"""Delete a layer to improve the above network, can train on Cifar10. Only output the class definition with its methods."""',
        'path':  './database\\network_14.py',
        'score': 429.9943981823001 
    },
    {
        'net': "",
        'prompt': '"""Add pooling layer to improve the above network, can train on Cifar10. Only output the class definition with its methods."""',
        'path': './database\\network_15.py',
        'score': 353.64286138561494
    },
    {
        'net': "",
        'prompt': '"""Add pooling layer to improve the above network, can train on Cifar10. Only output the class definition with its methods."""',
        'path': './database\\network_10.py',
        'score': 127.4588
    }
]

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



def tournament_selection(population, max_pop_size, tournament_size):
    """Perform tournament selection on the population

    Args:
        population (List[Dict]): The current population of individuals.
        max_pop_size (int): The number of individuals to retain in the next generation.
        tournament_size (int): The number of individuals in each tournament.

    Returns:
        List[Dict]: The new population after tournament selection.
    """
    selected_population = []
    unselected_individuals = population[:]  # Sao chép danh sách ban đầu

    while len(selected_population) < max_pop_size:
        # Lấy ngẫu nhiên một nhóm để đấu loại
        tournament = random.sample(unselected_individuals, tournament_size)
        # Chọn cá thể tốt nhất
        winner = max(tournament, key=lambda x: x["score"])
        selected_population.append(winner)
        # Loại bỏ cá thể chiến thắng khỏi danh sách các cá thể chưa được chọn
        unselected_individuals.remove(winner)

    # Xử lý các cá thể không được chọn
    for ind in unselected_individuals:
        ind_path = ind.get("path")
        if ind_path and os.path.exists(ind_path):
            try:
                os.remove(ind_path)
            except Exception as e:
                logging.error(f"Failed to remove {ind_path}: {e}")
        else:
            logging.warning(f"{ind_path} does not exist.")
    
    return selected_population

new = elitism_selection(
    population= networks_metadata,
    max_pop_size= 3, 
)
print(0)