# %%
import os
from collections import deque
from typing import Dict, List

os.chdir("/home")
import json

from vllm import SamplingParams

from PRMDataGeneration.omegaPRM import OmegaPRM, check_correctness
from Utils.ModelInterface import LLMInterface


# %%
def get_data(path: str) -> List[Dict]:
    """_summary_
        A dataset in  [{"problem:...,"final_answer":...},... ] format
    Args:
        path (str): data path

    Returns:
        Dict: dataset in [{"problem:...,"final_answer":...},... ]
    """
    with open(path, "rb") as f:
        data = json.load(f)
    return data


def data_filter(
    llm: LLMInterface, question: str, ground_truth_answer: str, num: int
) -> bool:
    """_summary_
        A data filter to eliminate too hard and too easy samples
    Args:
        llm (LLMInterface): LLM interface to generate rollout samples
        question (str): question
        ground_truth_answer (str): final answer
        num (int): number of rollouts

    Returns:
        bool: keep or not
    """
    pred_answers = llm.generate_rollout(question, num)
    total_correct_num = sum(
        [check_correctness(ans, ground_truth_answer) for ans in pred_answers]
    )
    return not (total_correct_num == 0 or total_correct_num == num)


def reformat_output(output: Dict) -> Dict:
    """_summary_
        Covert tree structure to  1 questio and multiple procress
    Args:
        output (Dict): output from OmegaPRM

    Returns:
        Dict: {"question": output["text"], "processed": [{'text':...,'label':...},...]}
    """
    collect = {"question": output["text"], "processed": []}
    queue = deque([["", child] for child in output["children"]])
    while queue:
        prev_text, children = queue.popleft()
        curr_text = prev_text + children["text"]
        collect["processed"].append(
            {
                "text": curr_text,
                "label": "+" if children["mc_value"] >= 0.5 else "-",
            }
        )
        for child in children["children"]:
            queue.append([curr_text, child])
    return collect


# %%
if __name__ == "__main__":
    llm = LLMInterface(
        "microsoft/Phi-4-mini-instruct",
        SamplingParams(max_tokens=2000, temperature=1.0, top_k=30),
    )
    # %%
    data = get_data("/home/MATH_problem_answer.json")[1]
    omega_prm = OmegaPRM(
        llm,
        c_puct=0.125,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=16,
        N=20,
        rollout_budget=100,
        save_data_tree=True,
    )
    reasoning_outputs = omega_prm.run(data["problem"], data["final_answer"])
    collected_data = {
        "question": data["problem"],
        "final_answer": data["final_answer"],
        "reasoning_steps": reformat_output(reasoning_outputs),
    }
    # %%
