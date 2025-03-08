# %%
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from GenerationMethod.beam_search import BeamSearchConfig, run_beam_search
from Utils.ModelInterface import PRMConfig, PRMInterface
from Utils.utils import check_correctness

if __name__ == "__main__":
    llm = LLM(
        model="microsoft/Phi-4-mini-instruct",
        trust_remote_code=True,
        max_model_len=6000,
        gpu_memory_utilization=0.5,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
    )

    model_path = "/home/models--openreasoner--Math-psa/snapshots/549f4cd6f4f75367f8466ac53bcdb7532897b392/checkpoint-2127"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    prm = PRMInterface(model, tokenizer, config=PRMConfig())

    with open("/home/test.jsonl", "rb") as file:
        file = [json.loads(f) for f in file]

    system_prompt = f"""Solve the following problem efficiently and clearly.
    Think step by step before answering.

    Follow these instructions carefully:
    - Structure your solution step by step, using '\n\n' to separate each step.
    - The start of each step should start with step i. i is the number of this step, example step1,step2
    - Your final answer should be formatted as follows:
    'Therefore, the final answer is: $\\boxed{{ANSWER}}$'

    Question:\n\n"""
    config = BeamSearchConfig(n=32, top_k=5)
    correct, incorrect = [], []
    for idx in range(500):
        prompt = file[idx]["problem"].replace("\n\n", "\n")
        generate_outputs = run_beam_search(prompt, config, system_prompt, llm, prm)
        if check_correctness(generate_outputs[0].current_text, file[idx]["answer"]):
            correct.append([file[idx], generate_outputs[0].current_text])
        else:
            incorrect.append([file[idx], generate_outputs[0].current_text])
        print(f"{len(correct)}/{len(correct)+len(incorrect)}")
