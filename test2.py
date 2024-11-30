import random 
from config import DEVICE
from config.prompts import PROMPT
from config.evolutionary import EAConfig
from operators.mutation import Mutation 
from operators.crossover import Crossover
from operators.base_operator import BaseConfig 
from correlation.foresight.pruners import predictive
from correlation.foresight.dataset import get_cifar_dataloaders
from utils.utils import *


MODEL_SUFFIX = os.environ.get("MODEL_SUFFIX")
database_net_path = "./database"

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
    }
]

def validate_and_check_network(generated_code, path, prompt, device):
    """Save network in file and check the network is trainable"""
    code_string = extract_code_section(generated_code)
    if not code_string:
        return None 

    write_codestring_to_file(code_string, path)
    try: 
        Net = get_class(path)
        net = create_instance(Net, in_channels= 3, out_channels= 64)
    except Exception as e: 
        logging.error(f"Error: {e}")
        os.remove(path)
        return None 

    if is_trainable(net):
        total_params =  sum(p.numel() for p in net.parameters())
        print(f"Total parameters: {total_params}")
    
        train_loader, _ = get_cifar_dataloaders(
            train_batch_size= 64, 
            test_batch_size= 64, 
            dataset= "cifar10",
            num_workers= 2, 
            resize= None 
        )

        score = predictive.find_measures(
            net_orig= net, 
            dataloader= train_loader, 
            dataload_info= ("random", 1, 10),
            device= device
        )
        return {
            "net": net, 
            "prompt": prompt, 
            "path": path, 
            "score": score["synflow"]
        }
    else: 
        os.remove(path)
        return None 

ranked_networks = ranking_individuals_in_pop(networks_metadata)
evo_operator = random.choices(["mutation", "crossover"], weights= [0.85, 0.15])[0]

network_id = 100
probabilities = [1.0 / len(PROMPT)] * len(PROMPT)
if evo_operator == "mutation":
    mutation_prompts =  random.choices(
        PROMPT, weights=probabilities, k=EAConfig.NUM_NET_MUTATION
    )
    for prompt in mutation_prompts: 
        network_id += 1 
        individual_mutations = select_individuals_mutation(ranked_networks)
        
        config_mutation = BaseConfig(
            model_name= MODEL_SUFFIX, 
            num_offspring= 1, 
            temperature= 0.7
        )

        mutation = Mutation(config_mutation)
        mutated_offspring = mutation._act(
            code_path= individual_mutations.get("path"),
            prompt= prompt 
        )[0]

        network_path = os.path.join(database_net_path, f"network_{network_id}.py")
        network_data = validate_and_check_network(
            generated_code= mutated_offspring, 
            path= network_path, 
            prompt= prompt, 
            device= DEVICE
        )
        if network_data: 
            networks_metadata.append(network_data)


elif evo_operator == "crossover":
    network_id += 1 

    config_crossover = BaseConfig(
        model_name= MODEL_SUFFIX, 
        num_offspring= 1, 
        temperature= 0.7
    )  

    crossover = Crossover(config_crossover)
    crossover_offsping = crossover._act(
        parent1_path= ranked_networks[0].get("path"),
        parent2_path= ranked_networks[1].get("path")
    )[0]
    network_path = os.path.join(database_net_path, f"network_{network_id}.py")
    network_data = validate_and_check_network(
        generated_code= crossover_offsping, 
        path= network_path, 
        prompt= "crossover_prompt",
        device= DEVICE
    )
    
    if network_data: 
        networks_metadata.append(network_data)

print(len(networks_metadata))