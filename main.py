import os
import sys
import random
import logging
import numpy as np
from config.prompts import PROMPT
from config.evolutionary import EAConfig
from operators.mutation import Mutation 
from operators.crossover import Crossover
from operators.base_operator import BaseConfig 
from generators.codegen import CodeGenerator
from correlation.foresight.pruners import predictive
from correlation.foresight.dataset import get_cifar_dataloaders
from config import *
from utils.utils import *


database_net_path = "./database"
if not os.path.exists(database_net_path):
    os.makedirs(database_net_path)

MODEL_SUFFIX = os.environ.get("MODEL_SUFFIX")
INIT_NET_PATH = os.environ.get("INIT_NET_PATH")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

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

def main():
    model_path = f"Salesforce/{MODEL_SUFFIX}"
    codegen = CodeGenerator(model_path)
    tokenizer, model = codegen.load_llm()

    networks_metadata = []
    network_id = 0 

    for generation in EAConfig.NUM_GENERATION:
        while len(networks_metadata) < 5:
            probabilities = [1.0 / len(PROMPT)] * len(PROMPT)

            selected_prompts = random.choices(
                PROMPT, weights= probabilities, k= EAConfig.NUM_NET_PER_GENERATION
            ) 
            init_net = read_python_file(INIT_NET_PATH)
            for prompt in selected_prompts:
                generated_net = codegen.inference(
                    prompt=init_net + "\n" + prompt,
                    temperature=0.5,
                    tokenizer=tokenizer,
                    llm_model=model,
                )
            
                # create path to save file 
                network_path = os.path.join(database_net_path, f"network_{network_id}.py")
                network_data = validate_and_check_network(
                    generated_code= generated_net, 
                    path= network_path,
                    prompt= prompt, 
                    device= DEVICE
                )

                if network_data: 
                    networks_metadata.append(network_path)
                network_id += 1 

    # Crossover / Mutation 
    ranked_networks = ranking_individuals_in_pop(networks_metadata)
    evo_operator = random.choices(["mutation", "crossover"], weights= [0.85, 0.15])[0]

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
            )

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
            parent1_path= ranked_networks[0],
            parent2_path= ranked_networks[1]
        )
        network_path = os.path.join(database_net_path, f"network_{network_id}.py")
        network_data = validate_and_check_network(
            generated_code= crossover_offsping, 
            path= network_path, 
            prompt= "crossover_prompt",
            device= DEVICE
        )

        if network_data: 
            networks_metadata.append(network_data)

    return networks_metadata


if __name__== "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    logging.basicConfig(
        level= logging.INFO, 
        format= '%(asctime)s , %(message)s', 
        datefmt='%Y%m%d%H%M',
        handlers=[
            logging.FileHandler("logs/logs.txt", encoding='utf-8'),  # write logs 
            logging.StreamHandler(stream= sys.stdout)  
        ], 
    )
    result = main()
