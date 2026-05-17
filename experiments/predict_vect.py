import csv
import pandas as pd
import os
import hydra

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from dotenv import load_dotenv
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))

    df = pd.read_csv(cfg.data_path)

    llm = LLM(
        model=cfg.llm.params.model,
        tensor_parallel_size=cfg.llm.params.tensor_parallel_size,
        max_model_len=cfg.llm.params.max_model_len,
        max_num_batched_tokens=cfg.llm.params.max_num_batched_tokens,
        dtype=cfg.llm.params.dtype
    )

    system_prompt = cfg.prompt.system.format(dim_adverb=cfg.dim.dim_adverb)
        
    guided_decoding_params = GuidedDecodingParams(choice=cfg.prompt.output_labels)
    sampling_params = SamplingParams(
        temperature=cfg.llm.params.temperature, # While it is a model parameter, all runs should use the same one.
        max_tokens=cfg.llm.params.max_tokens,
        guided_decoding=guided_decoding_params
    )

    prompts = []
    for _, row in df.iterrows():
        a = row["Premise A"]
        b = row["Premise B"]
        user_prompt = cfg.prompt.user.format(a=a, b=b)  
        
        if cfg.prompt.type == "few_shot":
            topic = row["Topic ID A"]
            system_prompt.format(
                ex_A_argA=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_A_argA"],
                ex_A_argB=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_A_argB"],
                ex_B_argA=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_B_argA"],
                ex_B_argB=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_B_argB"],
                ex_tie_argA=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_tie_argA"],
                ex_tie_argB=cfg.prompt.examples[f"tid_{topic}"][cfg.dim.name]["ex_tie_argB"],
            )
        
        prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

    resp = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
    )
        
    reply = [r.outputs[0].text.strip() for r in resp]

    df[f"Pred Comparison {cfg.dim.name.capitalize()}"] = reply

    filename = f"{cfg.llm.name}_{cfg.dim.name}_{cfg.prompt.type}_{cfg.run.run_id}.csv"
    df.to_csv(f"../gens/{filename}", index=False)

if __name__ == "__main__":
    main()
