import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenForCausalLM, BitsAndBytesConfig
import openai
import os
import ray
from transformers import AutoTokenizer, AutoModelForCausalLM, CodeGenForCausalLM

from conf.config import Config

cfg = Config()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def codex_mutate(cfg: Config, prompt, model="code-davinci-002", temperature=0.5):

    os.environ["OPENAI_API_KEY"] = ""  # ENETR OPENAI API KEY HERE
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=300,
        temperature=temperature,
    )


# ['codegen-350M-multi', 'codegen-2B-multi', 'codegen-6B-multi', 'codegen-16B-multi', 'codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'codegen-16B-mono']
# @ray.remote(num_cpus=cfg.NUM_CPUS)  # FIXME: Number of parallel processes should be configured globally.
@ray.remote(num_gpus=cfg.NUM_GPUS)
def codegen_mutate(cfg: Config, prompt, temperature):
    diff = False

    if diff:
        # model = "/mnt/lustre/users/mnasir/NAS-LLM/diff-codegen-6B"
        model = os.path.join(cfg.SAVE_DIR, "diff-codegen-6B")
        # tokenizer = AutoTokenizer.from_pretrained(model)#.to(cfg.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model)  # .to(cfg.DEVICE)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = CodeGenForCausalLM.from_pretrained(model).to(cfg.DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)  # .to(device)
        # model.config.use_cache = True
        sample = model.generate(
            **inputs, temperature=0.5, max_length=len(inputs[0]) + 300
        )

        return tokenizer.decode(
            sample[0][len(inputs[0]) :], clean_up_tokenization_spaces=True
        )
    else:
        model = f"Salesforce/{cfg.MUTATION}"
        print(f"Generate model with input prompt: {prompt}")
        # model = "/mnt/lustre/users/mnasir/NAS-LLM/codegen-6B"
        # model = "/mnt/lustre/users/mnasir/NAS-LLM/diff-codegen-6B"
        # model = os.path.join(cfg.SAVE_DIR, "codegen-6B")

        # TODO: Should be doing something like this to download the model automatically.
        # model = os.path.join(cfg.SAVE_DIR, cfg.MUTATION)
        # if not os.path.exists(model):
        #     # TODO: `Salesforce` part should not be hardcoded / should be configurable so that we can download models
        #     # from other sources.
        #     model = f'Salesforce/{cfg.MUTATION}'

        tokenizer = AutoTokenizer.from_pretrained(model)  # download tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto"
        )  # download model
        model.cuda()
        # TODO: Above lines may have downloaded a fresh model if it was not already present. Now, copy the model file
        # to the desired location if necessary.

        inputs = tokenizer(prompt, return_tensors="pt").to(cfg.DEVICE)  # encode prompt

        # generate code
        """generate code  
        input: encoded prompts 
        max_lenght: maximum length of result (350)
        temperature: control randomly
        do_sample: generate sample randomly instead of get best. 
        """
        sample = model.generate(
            **inputs,
            max_length=350 + len(inputs[0]),
            temperature=temperature,
            num_beams=1,
            do_sample=True,
        )

        # decode sample, remove backspace and return generated model architecture.
        return tokenizer.decode(
            sample[0][len(inputs[0]) :],
            truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
            clean_up_tokenization_spaces=True,
        )


def replace_word_mutation(sentence):
    if "Add" in sentence:
        return sentence.replace("Add", "Delete")
    elif "Delete" in sentence:
        return sentence.replace("Delete", "Add")
    else:
        return sentence
