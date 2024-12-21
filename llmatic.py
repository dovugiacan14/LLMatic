import os 
import sys
import torch 
import random
import logging 
import argparse
import numpy as np 

from config import DEVICE
from config.prompts import PROMPT
from utils.utils import *
from utils.network_validation import is_trainable

from fvcore.nn import FlopCountAnalysis
from generators.codegen import CodeGenerator
from map_elites import common as cm
from sklearn.neighbors import KDTree

from correlation.foresight.pruners import predictive
from correlation.foresight.dataset import get_cifar_dataloaders

database_net_path = "./database"
if not os.path.exists(database_net_path):
    os.makedirs(database_net_path)

MODEL_SUFFIX = os.environ.get("MODEL_SUFFIX")
INIT_NET_PATH = os.environ.get("INIT_NET_PATH")

train_loader, _ = get_cifar_dataloaders(
    train_batch_size= 64, 
    test_batch_size= 64, 
    dataset= "cifar10",
    num_workers= 2, 
    resize= None 
)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description= "Evolutionary Algorithm Configuration.")
    parser.add_argument("--num_generation", type= int, default= 2, help= "Number of generations")
    parser.add_argument("--pop_size", type= int, default= 5, help= "The minimum number of individuals required for the initial generation.")
    parser.add_argument("--max_pop_size", type= int, default= 5, help= "The maximum individuals in population.")
    parser.add_argument("--num_net", type= int, default= 10, help= "Number of network is generated per generation.")
    parser.add_argument("--num_mutate", type= int, default= 10, help= "'Number of networks to mutate.")
    parser.add_argument("--num_crossover", type= int, default= 10, help= "Number of networks to crossover.")
    parser.add_argument("--random_init_net", type= int, default= 10)
    parser.add_argument("--roll_out", type= int, default= 4, help= "For GPU training. ROLL_OUTS * (INIT_NUM_NETS or NUM_NETS) = Total nets created in each generation")
    parser.add_argument("--temperature_start", type= int, default= 0.7)
    parser.add_argument("--n_niches", type= int, default= 100, help= "Number of niches for map elites.")
    parser.add_argument("--save_dir", type= str, default= './')
    return parser.parse_args()


def to_specie(net, desc, fit,  net_path):
    return cm.Species(net, desc, fit, net_path)

def __add_to_archive(s, centroid, archive, kdt, type_ind):
    niche_index = kdt.query([centroid], k= 1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if type_ind == "network":
            if s.fitness < archive[n].fitness:
                archive[n] = s
                return True 
        elif type_ind == "prompt": 
            if s.fitness > archive[n].fitness: 
                archive[n] = s
                return True 
        return False 
    else: 
        archive[n] = s
        return True 
 
 
def main(args):
    # initialize codegen model 
    model_path = f"Salesforce/{MODEL_SUFFIX}"
    codegen = CodeGenerator(model_path) 
    tokenizer, model = codegen.load_llm()

    # read init net 
    init_net = read_python_file(INIT_NET_PATH)

    #curios_prompt = PROMPT[0]
    int_to_prompt = {}
    prompt_to_int = {}
    for i, prompt in enumerate(PROMPT):
        prompt_to_int[prompt] = i
        int_to_prompt[i] = prompt

    net_archive = {}
    prompt_archive = {}
    probabilities = [1.0 / len(PROMPT)] * len(PROMPT)

    prev_best_score = -np.inf
    exp_name = f"gen-nets_{MODEL_SUFFIX}_networks-{args.num_net}_niches-{args.n_niches}_infer-and-flops-as-bd"
    path_nets = f"{args.SAVE_DIR}/logs/{exp_name}"
    os.makedirs(path_nets, exist_ok=True)
    log_file = open(os.path.normpath(f"{path_nets}/cvt.dat"), "w")
    out_file = os.path.normpath(f"{path_nets}/exp_results.csv")
    csv_writer(["generations", "best_loss", "used_prompt"], out_file)

    # process for map elites. 
    params = cm.default_params
    c = cm.cvt(
        k = args.n_niches,
        dim= 2, 
        samples= params.get("cvt_samples")
    )

    kdt = KDTree(c, leaf_size= 30, metric= "euclidean")
    cm.__write_centroids(c)

    for gen_i in range(args.num_generation):
        print(f"======= GENERATION: {gen_i} ============")

        # Step 1: Generate Network 
        generated_nets = []
        if len(net_archive.keys()) < args.random_init_net and len(prompt_archive.keys()) < args.random_init_net * 2: 
            for _ in range(args.roll_out):
                selected_prompts = random.choices(PROMPT, weights= probabilities, k= args.num_net)
                for i in range(args.num_net):
                    print(f"Selected prompt for generation: {selected_prompts[i]}")
                
                for prompt in selected_prompts:
                    generated_net = codegen.inference(
                        prompt= init_net + "\n" + prompt, 
                        temperature= args.temperature_start,
                        tokenizer= tokenizer, 
                        llm_model= model
                    )
                    generated_nets.append((generated_net, prompt, args.temperature_start))

        else:  # variation or seclection 
            evo_operator = random.choices(["mutation", "crossover"], weights= [0.85, 0.15])[0]
            print(f"Performing {evo_operator}")
            for _ in range(args.roll_out):
                if evo_operator == "mutation":
                    if len(net_archive.keys()) < 3:
                        n_nets = 1 
                        selection = 0
                    else: 
                        n_nets = 3 
                        selection = random.randint(0,2)
                    
                    curios_nets = []
                    curios_prompts = []
                    curios_temps = []
                    curios_net_paths = []

                    n_curios_nets = get_max_fitness(net_archive.values(), n_nets)
                    for n in n_curios_nets:
                        curios_nets.append(n.x)
                        curios_net_paths.append(n.net_path)

                    n_prompts_temps = get_max_curiosity(prompt_archive.values(), n_nets)
                    for pt in n_prompts_temps:
                        curios_prompts.append(pt.desc[0])
                        curios_temps.append(pt.desc[1])
                    
                    curios_temp = curios_temps[selection]
                    curios_net_path = curios_net_paths[selection]

                    curios_temp_ray = []
                    curios_prompt_ray = []
                    for i in range(args.num_net):
                        # choose temperature to mutation 
                        if i > 1: 
                            curios_temp += random.uniform(-0.1, 0.1)
                            if curios_temp > 1.0: 
                                curios_temp = 1.0 
                            elif curios_temp < 0.1: 
                                curios_temp = 0.1 
                        
                        curios_prompt = curios_prompts[selection]
                        curios_temp_ray.append(curios_temp)
                        curios_prompt_ray.append(curios_prompt)
                    
                    for i in range(args.num_net):
                        generated_net_mutation = codegen.inference(
                            prompt= read_python_file(curios_net_path) + "\n" + int_to_prompt[int(curios_prompt_ray[i])],
                            temperature= curios_temp_ray[i], 
                            tokenizer= tokenizer, 
                            llm_model= model
                        )
                        generated_nets.append(
                            generated_net_mutation, 
                            int_to_prompt[int(curios_prompt_ray[i])], 
                            curios_temp_ray[i]
                        )

                elif evo_operator == "crossover":
                    if len(net_archive.keys()) < 1: 
                        print("Can't perform crossover")
                    else:
                        if len(net_archive.keys()) < 2: 
                            selection = 0 
                            n_nets= 1 
                        else: 
                            selection = random.randint(
                                1, len(net_archive.keys()) - 1
                            )
                            n_nets = 2 

                    curios_net_paths = []
                    curios_prompts = []
                    curios_temp = []
                    curios_nets_ray = []
                    
                    n_curios_nets = get_max_fitness(net_archive.values(), n_nets)
                    for n in n_curios_nets: 
                        curios_nets.append(n.x)
                        curios_net_paths.append(n.net_path)
                    
                    n_prompts_temps = get_max_curiosity(prompt_archive.values(), n_nets)
                    for pt in n_prompts_temps:
                        curios_prompts.append(pt.desc[0])
                        curios_temps.append(pt.desc[1])
                    
                    curios_net_str = read_python_file(curios_net_paths[0]) 
                    curios_temp = curios_temps[0]
                    curios_prompt = curios_prompts[0]
                    for i in range(0, args.num_net):
                      
                        curios_nets_ray.append(
                            read_python_file(curios_net_paths[selection])
                        )

                    crossover_prompt = '"""Combine the above two neural networks and create a third neural network class that also inherits from nn.Module"""'

                    for curious_2nd_net in curios_nets_ray: 
                        generated_net_crossover = codegen.inference(
                            prompt = curios_net_str + "\n" + curious_2nd_net + "\n" + crossover_prompt,
                            temperature= curios_temp, 
                            tokenizer= tokenizer, 
                            llm_model= model
                        )
                        generated_nets.append(generated_net_crossover, int_to_prompt[int(curios_prompt)], curios_temp)

        # Step 2: extract class from codestring and check is_trainable 
        net_paths = []
        training_prompts = []
        training_nets = []
        for idx, net_item in enumerate(generated_nets):
            generation, prompt, temperature = net_item
            net_path = os.path.normpath(f"{database_net_path}/network_gen{gen_i}_{idx}.py")

            code_string = extract_code_section(generation)
            if not code_string: 
                code_string = " "
            write_codestring_to_file(code_string, net_path)
            main_net_focus = read_python_file(net_path)
            
            try: 
                Net = get_class(net_path)
                # net = Net()
                net = create_instance(Net, in_channels= 3, out_channels= 64)
            except Exception:
                print(f"The network at {net_path} is invalid.")
                continue
                # Net = get_class(init_net)
                # net_path = INIT_NET_PATH
                # if isinstance(curios_prompt, str):
                #     prompt = curios_prompt
                # elif isinstance(curios_prompt, float) or isinstance(curios_prompt, int):
                #     prompt = int_to_prompt[int(curios_prompt)]

            if is_trainable(net):
                net_paths.append(net_path)
                training_prompts.append(prompt)
                training_nets.append(net)
                
            else:
                print(f"The network at {net_path} is not trainable")

        if training_nets == []:
            continue 
        
        # Step 3: calculate fitness based on synflow score 
        # if len(net_archive.keys()) < args.random_init_net and len(prompt_archive.keys()) < args.random_init_net * 2:
        inter_results = []
        fitness = []
        for net in training_nets:
            score = predictive.find_measures(
                net_orig= net, 
                dataloader= train_loader, 
                dataload_info= ("random", 1, 10),
                device= DEVICE
            ).get("synflow", 0)
            inter_results.append(score)
        
        for i, result in enumerate(inter_results):
            try:
                fitness.append(
                    [
                        result, 
                        training_prompts[i],
                        temperature,
                        net_paths[i]
                    ]
                )
                if fitness[i][0] >= prev_best_score:
                    prev_best_score = fitness[i][0]
                    temperature += 0.05
                else: 
                    temperature -= 0.05 
                if temperature > 1.0: 
                    temperature = 1.0 
                elif temperature < 0.1: 
                    temperature = 0.1 
                fitness[i][2] = temperature
            except Exception: 
                print("Not trainable due to fitness 1.")
        
        # Step 4: Store to net archive and prompt archive
        dummy_input = torch.zeros((1, 3, 32, 32))
        for score_x, prompt_x, temp_x, net_p in fitness: 
            Net = get_class(net_p)
            net = Net()
            try:
                flps = FlopCountAnalysis(net, dummy_input)
                flops= flps.total()
                depth_with_ratio = get_network_width_depth_ratio(net)
            except Exception: 
                flops = 0 
                depth_with_ratio = 1
            
            print(f"Flops: {flops}; Depth-Width-Ratio: {depth_with_ratio}")

            # process net archive 
            s_net = to_specie(
                net= net, 
                desc= np.array([depth_with_ratio, flops]),
                fit= score_x, 
                net_path= net_p
            )
            net_added = __add_to_archive(
                s= s_net, 
                centroid= s_net.desc, 
                archive= net_archive, 
                kdt= kdt, 
                type_ind= "network"
            )

            if net_added:
                prompt_fit = 1.0 
            else: 
                prompt_fit = 0.0

            # process prompt archive 
            s_prompt = to_specie(
                net= net, 
                desc= np.array([prompt_to_int[prompt_x], temp_x]),
                fit= prompt_fit, 
                net_path= net_p
            )
            prompt_added = __add_to_archive(
                s= s_prompt, 
                centroid= s_prompt.desc, 
                archive= prompt_archive, 
                kdt= kdt, 
                type_ind= "prompt"
            )
            if not net_added:
                s_prompt.curiosity = s_prompt - 0.5 
            elif net_added: 
                s_prompt.curiosity = s_prompt + 1.0 
        if gen_i % i == 0:
            cm.__save_archieve(net_archive, gen_i, name= "net")
            cm.__save_archieve(prompt_archive, gen_i, name= "prompt")

        if log_file != None: 
            fit_list = np.array([x.fitness for x in net_archive.values()])
            log_file.write(
                "{} {} {} {} {} {} {}\n".format(
                    gen_i,
                    len(net_archive.keys()),
                    len(prompt_archive.keys()),
                    fit_list.min(),
                    np.mean(fit_list),
                    np.median(fit_list),
                    np.percentile(fit_list, 5),
                    np.percentile(fit_list, 95),
                )
            )
            log_file.flush()
    cm.__save_archieve(net_archive, gen_i, name= "net")
    cm.__save_archieve(prompt_archive, gen_i, name= "prompt")
    return net_archive, prompt_archive

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

    args = parse_arguments()
    result = main(args)
