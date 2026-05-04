import csv
import time
from vllm import LLM, SamplingParams
from collections import defaultdict
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
import hydra


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

# NUM_ARGUMENTS =  750      # Number of arguments to sample from the dataset       
#SLEEP_EVERY = 100              
#SLEEP_DURATION = 10  
#          
# Prompt templates for each quality dimension

PROMPTS = {
    "logic": {
        "system": (
            "You are given two arguments: Argument A and Argument B.\n"
            "Decide which one is logically stronger based on these criteria only:\n"
            "- which is more acceptable/credible \n"
            "- which is more relevant to a conclusion\n"
            "- which is more sufficient to justify a conclusion\n"
            "Reply with only one of the following options: A, B, or tie. Do NOT add any other text.\n"
        ),
        "user": "Argument A: {a}\nArgument B: {b}"
    },
    "rhetoric": {
        "system": (
            "You are given two arguments: Argument A and Argument B.\n"
            "Decide which one is rhetorically stronger based on these criteria only:\n"
            "- which appears more authorative/trust worthy\n"
            "- which makes a stronger emotional appeal\n"
            "- which is clear and more appriopriate in tone\n"
            "Reply with only one of the following options: A, B, or tie. Do NOT add any other text.\n"
        ),
        "user": "Argument A: {a}\nArgument B: {b}"
    }, 
    "dialectic": {
        "system": (
            "You are given two arguments: Argument A and Argument B.\n"
            "Decide which one is dialectically stronger based on these criteria only:\n"
            "- which would be acceptable to the audience\n"
            "- which contributes more to constructive dialogue\n"
            "- which better anticipates or refutes counterarguments\n"
            "Reply with only one of the following options: A, B, or tie. Do not provide any explanation.\n"
        ), 
        "user": "Argument A: {a}\nArgument B: {b}"
    }
}


# Group sampled arguments by topic, and track their original list index
def group_args_by_topic_with_indices(sampled_args):
    grouped = defaultdict(list)
    for idx, arg in enumerate(sampled_args):
        arg['global_index'] = idx  # track position in original list
        grouped[arg['topic_id']].append(arg)
    return grouped

# Generate cyclic pairs within each topic group
def create_cyclic_pairs_within_topics(grouped_args, step=5):
    all_pairs = []
    for topic_id, args in grouped_args.items():
        n = len(args)
        if n < 2:
            continue  # skip topics with only one argument
        for i in range(n):
            for j in range(1, step + 1):
                a = args[i]['global_index']
                b = args[(i + j) % n]['global_index']
                pair = tuple(sorted((a, b)))
                if pair not in all_pairs:
                    all_pairs.append(pair)
    return all_pairs


# Function to send a pair of arguments to the Together AI API and get a comparison result
def comparisons(a, b, prompt_template, llm):
    #client = Together()  
    prompt = prompt_template.format(a=a['premise'], b=b['premise'])  
    #try:
    sampling_params = SamplingParams(
        temperature=0.0,
    )

    # Send prompt to the Together AI model
    resp = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=sampling_params,
        )
        #resp = client.chat.completions.create(
        #    model=MODEL_NAME,
        #    messages=[{"role": "user", "content": prompt}],
        #    temperature=0.0
        #)
        # Extract the model's response
    #print(resp)
    reply = resp[0].outputs[0].text.strip().lower()
    #reply = resp.outputs[0].text.strip().lower()
        #reply = resp.choices[0].message.content.strip().lower()

        # Return winner based on model's response
    if reply == "a":
        return "A"
    elif reply == "b":
        return "B"
    else:
        return "tie"
   # except Exception as e:
   #     print("API error:", e)
   #     return None

# Function to evaluate a batch of argument pairs for a specific dimension
def run_dimension_comparison(dimension, sampled_args, pairs, model_name, model_path, tensor_parallel_size, max_model_len, max_num_batched_tokens):
    filename = f"{model_name}_{dimension}_comparisons_v2.csv"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        #max_model_len=max_model_len,
        #max_num_batched_tokens=max_num_batched_tokens,
        #decode_context_parallel_size=1,
        dtype="auto",
    )
    prompt_template = PROMPTS[dimension]  

    # Open output file and set up CSV writer with discussion IDs included
    with open(filename, "w", newline='', encoding='utf-8-sig') as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            'Discussion ID A', 'Argument ID A',
            'Discussion ID B', 'Argument ID B',
            'Comparison'
        ])
        writer.writeheader() 

        sampling_params = SamplingParams(
            temperature=0.0,
        )
        # Loop over all sampled argument pairs
        prompts = []
        for _, (i, j) in enumerate(pairs):
            a = sampled_args[i]
            b = sampled_args[j]        
            prompt = prompt_template["user"].format(a=a['premise'], b=b['premise'])  
            prompts.append(prompt)

        # Send prompt to the Together AI model
        resp = llm.chat(
            messages=[{"role": "system", "content": prompt_template["system"]},
                      {"role": "user", "content": p} for p in prompts],
            sampling_params=sampling_params,
        )
        
        reply = [r.outputs[0].text.strip().lower() for r in resp]
        reply_enc = []
        for r in reply:
            if reply == "a":
                reply_enc.append("A")
            elif reply == "b":
                reply_enc.append("B")
            else:
                reply_enc.append("tie")

        for r in reply_enc:
            # Write the result to the CSV, including discussion IDs
            writer.writerow({
                'Discussion ID A': a['discussion_id'],
                'Argument ID A': a['argument_id'],
                'Discussion ID B': b['discussion_id'],
                'Argument ID B': b['argument_id'],
                'Comparison': r
            })

    print(f"Finished writing {dimension} results to {filename}\n")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    #sampled_df = pd.read_csv("sampled_args_reconstructed.csv")
    df = pd.read_csv(cfg.data_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Topic ID": "topic_id",
        "Discussion ID": "discussion_id",
        "Argument ID": "argument_id",
        "Premise": "premise"
    }, inplace=True)    
    args = df.to_dict(orient="records")

    model_name = cfg.llm.name
    model_path = cfg.llm.params.model
    tensor_parallel_size = cfg.llm.params.tensor_parallel_size
    max_model_len=cfg.llm.params.max_model_len
    max_num_batched_tokens=cfg.llm.params.max_num_batched_tokens

    # Group arguments by topic and generate cyclic pairs within topics
    grouped_args = group_args_by_topic_with_indices(args)
    pairs = create_cyclic_pairs_within_topics(grouped_args, step=5)

    print(f"Generated {len(pairs)} within-topic cyclic pairs.")

    # Loop through quality dimensions
    for dimension in PROMPTS.keys():
        run_dimension_comparison(dimension,
                                 args,
                                 pairs,
                                 model_name,
                                 model_path,
                                 tensor_parallel_size,
                                 max_model_len,
                                 max_num_batched_tokens
        )

if __name__ == "__main__":
    main()
