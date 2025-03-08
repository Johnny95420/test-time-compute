import re
from dataclasses import dataclass

import torch
from torch import nn
from vllm import LLM


class LLMInterface:
    def __init__(self, model_name: str, sampling_params, **kwargs):
        self.llm = LLM(
            model=model_name, trust_remote_code=True, max_model_len=40000, **kwargs
        )
        self.sampling_params = sampling_params
        self.instruction = """
Solve the following math problem efficiently and clearly. Think step by step before answering.

Follow these instructions carefully:
- Structure your solution step by step, using '\n\n' to separate each step.
- If the solution is already provided, do not regenerate the existing steps or other answers. Instead, simply end the conversation.
- Your final answer should be formatted as follows:  
'Therefore, the final answer is: $\\boxed{{ANSWER}}$'
(where ANSWER is the final numerical or algebraic solution, without quotes).

Question:\n\n
"""

    def generate_rollout(self, prefix, nums):
        outputs = self.llm.generate(
            [self.instruction + prefix] * nums, self.sampling_params
        )

        return [output.outputs[0].text for output in outputs]


@dataclass
class PRMConfig:
    good_token: str = "+"
    bad_token: str = "-"
    output_step_tag: str = "\n\n"
    model_step_tag: str = " \n\n\n\n\n"


class PRMInterface:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        self.config = config
        self.candidate_tokens = self.tokenizer.encode(
            f" {self.config.good_token} {self.config.bad_token}"
        )
        self.step_tag_token = self.tokenizer.encode(self.config.model_step_tag)[-1]

    def head_replace(self):
        new_head = nn.Linear(self.model.lm_head.in_features, 2).to(
            device=self.model.device, dtype=self.model.dtype
        )
        new_head.weight.data = self.model.lm_head.weight.data[self.candidate_tokens,]
        self.model.lm_head = new_head

    def score(self, prompts, steps):
        good_probs = []
        for prompt, step in zip(prompts, steps):
            output = f"{prompt} {step}".replace(
                self.config.output_step_tag, self.config.model_step_tag
            )
            output = re.sub(r"\s+\n\n\n\n\n", " \n\n\n\n\n", output)
            output = self.tokenizer.encode(output, return_tensors="pt").cuda()
            with torch.inference_mode(mode=True):
                logits = self.model(output).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:, :, 0]
                step_scores = scores[output == self.step_tag_token]
                good_probs.append(step_scores.float().cpu().numpy())
        return good_probs
