import os 
import json
import torch 
import random
import argparse
import numpy as np 
from utils.utils import *
from config import DEVICE
from config.prompts import PROMPT
from map_elites import common as cm
from sklearn.neighbors import KDTree
from fvcore.nn import FlopCountAnalysis
# from generators.codegen import CodeGenerator
from generators.qwen import CodeGenerator
from utils.network_validation import is_trainable
from correlation.foresight.pruners import predictive
from correlation.foresight.dataset import get_cifar_dataloaders


MODEL_SUFFIX = os.environ.get("MODEL_SUFFIX")
INIT_NET_PATH = os.environ.get("INIT_NET_PATH")
train_loader, _ = get_cifar_dataloaders(
    train_batch_size= 64, 
    test_batch_size= 64, 
    dataset= "cifar10",
    num_workers= 2, 
    resize= None 
)

database_net_path = "./database"
if not os.path.exists(database_net_path):
    os.makedirs(database_net_path)

prompt_to_int = {}
for i, prompt in enumerate(PROMPT):
    prompt_to_int[prompt] = i

def parse_arguments():
    parser = argparse.ArgumentParser(description= "Evolutionary Algorithm Configuration.")
    parser.add_argument("--num_generation", type= int, default= 2, help= "Number of generations")
    parser.add_argument("--pop_size", type= int, default= 5, help= "The minimum number of individuals required for the initial generation.")
    parser.add_argument("--max_pop_size", type= int, default= 5, help= "The maximum individuals in population.")
    parser.add_argument("--num_net", type= int, default= 2, help= "Number of network is generated per generation.")
    parser.add_argument("--num_mutate", type= int, default= 10, help= "'Number of networks to mutate.")
    parser.add_argument("--num_crossover", type= int, default= 10, help= "Number of networks to crossover.")
    parser.add_argument("--random_init_net", type= int, default= 3)
    parser.add_argument("--roll_out", type= int, default= 1, help= "For GPU training. ROLL_OUTS * (INIT_NUM_NETS or NUM_NETS) = Total nets created in each generation")
    parser.add_argument("--temperature_start", type= int, default= 0.7)
    parser.add_argument("--n_niches", type= int, default= 100, help= "Number of niches for map elites.")
    parser.add_argument("--save_dir", type= str, default= './')
    return parser.parse_args()

class NetworkGenerator:
    def __init__(self, model_path, init_net_path):
        self.codegen = CodeGenerator(model_path)
        self.tokenizer, self.model = self.codegen.load_llm()
        self.init_net = read_python_file(init_net_path)
        
    def generate_random(self, prompt, temperature):
        """Generate a new network using random prompt"""
        generated_net = self.codegen.inference(
            prompt=self.init_net + "\n" + prompt,
            temperature=temperature,
            tokenizer=self.tokenizer,
            llm_model=self.model
        )
        return generated_net

    def generate_mutation(self, net_path, prompt, temperature):
        """Generate a new network by mutating existing one"""
        base_net = read_python_file(net_path)
        generated_net = self.codegen.inference(
            prompt=base_net + "\n" + prompt,
            temperature=temperature,
            tokenizer=self.tokenizer,
            llm_model=self.model
        )
        return generated_net

    def generate_crossover(self, net_path1, net_path2, temperature):
        """Generate a new network by crossing over two networks"""
        net1 = read_python_file(net_path1)
        net2 = read_python_file(net_path2)
        crossover_prompt = '"""Combine the above two neural networks and create a third neural network class that also inherits from nn.Module"""'
        
        generated_net = self.codegen.inference(
            prompt=net1 + "\n" + net2 + "\n" + crossover_prompt,
            temperature=temperature,
            tokenizer=self.tokenizer,
            llm_model=self.model
        )
        return generated_net

class NetworkEvaluator:
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device
        self.dummy_input = torch.zeros((1, 3, 32, 32)).to(device)

    def validate_network(self, net_path):
        """Check if network is valid and trainable"""
        try:
            Net = get_class(net_path)
            net = create_instance(Net, in_channels=3, out_channels=64)
            return is_trainable(net), net
        except Exception:
            print(f"The network at {net_path} is invalid.")
            return False, None

    def calculate_fitness(self, net):
        """Calculate fitness score using synflow"""
        return predictive.find_measures(
            net_orig=net,
            dataloader=self.train_loader,
            dataload_info=("random", 1, 10),
            device=self.device
        ).get("synflow", 0)
        
    def calculate_descriptors(self, net):
        """Calculate network descriptors (FLOPs and depth/width ratio)"""
        try:
            flops = FlopCountAnalysis(net, self.dummy_input).total()
            depth_ratio = get_network_width_depth_ratio(net)
            return flops, depth_ratio
        except Exception as e:
            print(f"Can't calculate FLOPS score: {e}")
            return 0, 1
    
class ArchiveManager: 
    def __init__(self, n_niches): 
        self.n_niches = n_niches 
        self.net_archive = {}
        self.prompt_archive = {}
        self.kdt = self._initialize_kdtree()
        
    def _initialize_kdtree(self): 
        params = cm.default_params 
        centroids = cm.cvt(
            k= self.n_niches, 
            dim= 2, 
            samples= params.get("cvt_samples")
        )
        return KDTree(centroids, leaf_size= 30, metric= "euclidean")
    
    def _add_to_archive(self, species, centroid, archive, type_ind): 
        niche_index = self.kdt.query([centroid], k= 1)[1][0][0]
        niche = self.kdt.data[niche_index]
        n = cm.make_hashable(niche)
        species.centroid = n 

        if n in archive: 
            if type_ind == "network" and species.fitness < archive[n].fitness: 
                archive[n] = species 
                return True 
            elif type_ind == "prompt" and species.fitness > archive[n].fitness: 
                archive[n] = species 
                return True 
            return False 
        else: 
            archive[n] = species
            return True 

    def add_to_archives(self, net, net_path, descriptors, fitness_score, prompt, temperature): 
        """Add network and prompt to corresponding archives"""
        # add to net archive 
        s_net = cm.Species(net, np.array(descriptors), fitness_score, net_path)
        net_added = self._add_to_archive(
            s_net, 
            s_net.desc, 
            self.net_archive, 
            "network"
        )

        # add to prompt archive 
        prompt_fitness = 1.0 if net_added else 0.0 
        s_prompt = cm.Species(net, np.array([float(prompt_to_int[prompt]), temperature]), prompt_fitness, net_path)
        self._add_to_archive(s_prompt, s_prompt.desc, self.prompt_archive, "prompt")

        return net_added

    def save_archives(self, generation): 
        """Save current state of archives"""
        cm.save_archieve(self.net_archive, generation, name= "net")
        cm.save_archieve(self.prompt_archive, generation, name= "prompt")

def main(args): 
    # initialize components 
    # generator = NetworkGenerator(
    #     model_path= f"Salesforce/{MODEL_SUFFIX}",
    #     init_net_path= INIT_NET_PATH
    # )
    generator = NetworkGenerator(
        model_path= "Qwen/Qwen2.5-Coder-7B",
        init_net_path= INIT_NET_PATH
    )
    evaluator = NetworkEvaluator(train_loader, DEVICE)
    archive_manager = ArchiveManager(args.n_niches)
    
    temperature = args.temperature_start 
    prev_best_score = -np.inf

    # main evolution loop 
    for gen_i in range(args.num_generation): 
        print(f"======= GENERATION: {gen_i} ============")

        # step 1: generate networks 
        generated_nets = []
        if len(archive_manager.net_archive) < args.random_init_net: 
            # initial random generation 
            for prompt in random.choices(PROMPT, k= args.num_net): 
                net = generator.generate_random(prompt, temperature)
                generated_nets.append((net, prompt, temperature))
        else: 
            # evolution through mutation or crossover 
            if random.random() < 0.85: # mutation 
                nets = get_max_fitness(archive_manager.net_archive.values(), 3)
                for net in nets: 
                    mutated_net = generator.generate_mutation(net.net_path, net.prompt, temperature)
                    generated_nets.append((mutated_net, net.prompt, temperature))
            else:  # crossover 
                nets = get_max_fitness(archive_manager.net_archive.values(), 2)
                crossed_net = generator.generate_crossover(nets[0].net_path, nets[1].path, temperature)
                generated_nets.append((crossed_net, nets[0].prompt, temperature))
        
        # step 2&3: validate and evaluate networks 
        for idx, (net_code, prompt, temp) in enumerate(generated_nets):
            net_path = os.path.join(database_net_path,  f"network_gen{gen_i}_{idx}.py")
            write_codestring_to_file(extract_code_section(net_code), net_path)

            is_valid, net = evaluator.validate_network(net_path)
            if not is_valid: 
                continue 

            fitness = evaluator.calculate_fitness(net)
            flops, depth_ratio = evaluator.calculate_descriptors(net)

            # update temperature based on fitness 
            if fitness >= prev_best_score: 
                temperature = min(1.0, temperature + 0.05)
                prev_best_score = fitness 
            else: 
                temperature = max(0.1, temperature - 0.05)
            
            # step 4: update archives 
            archive_manager.add_to_archives(
                net= net, 
                net_path= net_path, 
                descriptors= [depth_ratio, flops], 
                fitness_score= fitness, 
                prompt= prompt, 
                temperature= temp
            )
        
        # save archives periodically 
        archive_manager.save_archives(gen_i)
    return archive_manager.net_archive, archive_manager.prompt_archive

if __name__ == "__main__": 
    args = parse_arguments()
    net_archive, prompt_archive = main(args)
    print("Done.!")


    


            
