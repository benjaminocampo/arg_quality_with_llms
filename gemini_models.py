import google.generativeai as genai
import csv
import time
import pandas as pd
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")

# Configuration
SLEEP_EVERY = 50
SLEEP_DURATION = 60
# 2.0 can't be used anymore
#MODEL_NAME = "gemini-2.5-flash-lite" 
MODEL_NAME = "gemini-2.5-flash"

# Authenticate Gemini API
genai.configure(api_key=GEMINI_TOKEN)

# Prompt Templates for Each Dimension
PROMPTS = {
    "logic": (
        "You are given two arguments: Argument A and Argument B.\n"
        "Decide which one is rhetorically stronger based on these criteria only:\n"
        "- which is more acceptable/credible \n"
        "- which is more relevant to a conclusion\n"
        "- which is more sufficient to justify a conclusion\n"
        "Reply with only one of the following options: A, B, or tie. Do NOT add any other text.\n"
        "Argument A: {a}\nArgument B: {b}"
    ),
    "rhetoric": (
        "You are given two arguments: Argument A and Argument B.\n"
        "Decide which one is rhetorically stronger based on these criteria only:\n"
        "- which appears more authorative/trust worthy\n"
        "- which makes a stronger emotional appeal\n"
        "- which is clear and more appriopriate in tone\n"
        "Reply with only one of the following options: A, B, or tie. Do NOT add any other text.\n"
        "Argument A: {a}\nArgument B: {b}"
    ),
    "dialectic": (
        "You are given two arguments: Argument A and Argument B.\n"
        "Decide which one is rhetorically stronger based on these criteria only:\n"
        "- which would be acceptable to the audience\n"
        "- which contributes more to constructive dialogue\n"
        "- which better anticipates or refutes counterarguments\n"
        "Reply with only one of the following options: A, B, or tie. Do not provide any explanation.\n"
        "Argument A: {a}\nArgument B: {b}"
    ),
}
GENERATION_CONFIG = {
    "temperature": 0.0
}

def group_args_by_topic_with_indices(sampled_args):
    grouped = defaultdict(list)
    for idx, arg in enumerate(sampled_args):
        arg['global_index'] = idx
        grouped[arg['topic_id']].append(arg)
    return grouped

def create_cyclic_pairs_within_topics(grouped_args, step=3):
    all_pairs = []
    for topic_id, args in grouped_args.items():
        n = len(args)
        if n < 2:
            continue
        for i in range(n):
            for j in range(1, step + 1):
                a = args[i]['global_index']
                b = args[(i + j) % n]['global_index']
                pair = tuple(sorted((a, b)))
                if pair not in all_pairs:
                    all_pairs.append(pair)
    return all_pairs

#  Perform LLM Comparison 
def comparisons(a, b, prompt_template):
    prompt = prompt_template.format(a=a['premise'], b=b['premise'])
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
        reply = response.text.strip().lower()

        if reply == "a":
            return "A"
        elif reply == "b":
            return "B"
        else:
            return "tie"
    except Exception as e:
        print("API error:", e)
        return None

#  Run Evaluation 
def run_dimension_comparison(dimension, sampled_args, sampled_pairs):
    #df_res = pd.read_csv(f"gemini_{dimension}_comparisons_v1.csv")
    filename = f"gemini_{dimension}_comparisons_v2.csv"
    prompt_template = PROMPTS[dimension]

    #null_comp = df_res["Comparison"].isna()
    #null_idx = df_res[null_comp].index
    #for idx, (i, j) in enumerate(sampled_pairs):
    #    if idx not in null_idx:
    #        continue
    #    a = sampled_args[i]
    #    b = sampled_args[j]
    #    result = comparisons(a, b, prompt_template)
    #    df_res.loc[idx, "Discussion ID A"] = a['discussion_id']
    #    df_res.loc[idx, "Argument ID A"] = a['argument_id']
    #    df_res.loc[idx, "Discussion ID B"] = b['discussion_id']
    #    df_res.loc[idx, "Argument ID B"] = b['argument_id']
    #    df_res.loc[idx, "Comparison"] = result

    #df_res.to_csv(f"gemini-2.5-lite_{dimension}_comparisons_v2.csv", index=False)
    with open(filename, "w", newline='', encoding='utf-8-sig') as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            'Discussion ID A', 'Argument ID A',
            'Discussion ID B', 'Argument ID B',
            'Comparison'
        ])
        writer.writeheader()
        
    
        for idx, (i, j) in enumerate(sampled_pairs):
            a = sampled_args[i]
            b = sampled_args[j]
            result = comparisons(a, b, prompt_template)
    
            writer.writerow({
                'Discussion ID A': a['discussion_id'],
                'Argument ID A': a['argument_id'],
                'Discussion ID B': b['discussion_id'],
                'Argument ID B': b['argument_id'],
                'Comparison': result
            })
    
            print(f"[{dimension.upper()}] Compared {idx+1}/{len(sampled_pairs)} - Result: {result}")
    
            if (idx + 1) % SLEEP_EVERY == 0:
                print("Break...")
                time.sleep(SLEEP_DURATION)

    print(f"Finished writing {dimension} results to {filename}\n")


# MAIN 
if __name__ == "__main__":
    sampled_df = pd.read_csv("sampled_args_reconstructed.csv")
    sampled_df.rename(columns={
        "Topic ID": "topic_id",
        "Discussion ID": "discussion_id",
        "Argument ID": "argument_id",
        "Premise": "premise"
    }, inplace=True)
    sampled_args = sampled_df.to_dict(orient="records")
    grouped_args = group_args_by_topic_with_indices(sampled_args)
    sampled_pairs = create_cyclic_pairs_within_topics(grouped_args, step=3)

    for dimension in PROMPTS.keys():
        run_dimension_comparison(dimension, sampled_args, sampled_pairs)
