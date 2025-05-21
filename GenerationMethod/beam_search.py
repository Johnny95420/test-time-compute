"""
Parts of this code are referenced from the sal project.
For more details, please refer to https://github.com/salopensource/sal.
"""

import json
import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from Utils.ModelInterface import PRMConfig, PRMInterface
from Utils.utils import check_correctness


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None


@dataclass
class BeamSearchConfig:
    n: int = 32
    beam_width: int = 4
    num_iterations: int = 20
    custom_chat_template: Optional[str] = None
    lookahead: int = 0  # lookahead step
    stop_word: List[str] = field(default_factory=lambda: ["\n\n", "<|end|>"])
    top_p: float = 0.95
    top_k: int = 5
    temperature: float = 1
    max_tokens: int = 500
    early_stop_criteria: Optional[int] = None


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None or gen_text.endswith(
                    "<|end|>"
                ):
                    gen_result.first_step_text += "\n\n"
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None or gen_text.endswith("<|end|>"):
                gen_result.lookahead_text += "\n\n"
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs


def create_initial_beams(
    prompt: List[str],
    config,
) -> List:
    # initial beam list
    beams = []
    for i in range(config.n):
        beams.append(
            Beam(
                prompt=prompt,
                index=i,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                pruned=False,
                completed=False,
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=0,
            )
        )
    return beams


def prepare_sampling_params(config: BeamSearchConfig) -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        top_k=config.top_k,
        stop=config.stop_word,
        include_stop_str_in_output=True,
        n=1,
    )
    return sampling_params


def adjust_active_beams(active_beams: List, config: BeamSearchConfig) -> List:
    """
    impute active beams to n
    """
    if len(active_beams) == config.n:
        return active_beams

    repeats = (config.n // len(active_beams)) + 1
    extended_active_beams = [
        copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
    ]
    if len(extended_active_beams) != config.n:
        raise ValueError(
            f"Expected {config.n} active beams, but got {len(extended_active_beams)}"
        )
    return extended_active_beams


def prepare_iteration_sampling_params(
    config: BeamSearchConfig, last_round: bool
) -> SamplingParams:
    if last_round:
        # * in the last round genenrate to the end
        stop_words = copy.deepcopy(config.stop_word)
        if "\n\n" in stop_words:
            stop_words.remove("\n\n")
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_k=config.top_k,
            n=1,
            stop=stop_words,
        )
    else:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=config.stop_word,
            include_stop_str_in_output=True,
            n=1,
        )
    return sampling_params


def generate_for_active_beams(
    active_beams: List,
    system_prompt: str,
    llm: torch.nn.Module,
    config: BeamSearchConfig,
    sampling_params: SamplingParams,
    iteration_index: int,
):
    """
    Generate and modify Beams
    """

    lookahead = 0 if iteration_index == config.num_iterations - 1 else config.lookahead
    continue_final_message = iteration_index > 0
    add_generation_prompt = iteration_index == 0

    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    convs = [build_conv(b.prompt, b.current_text, system_prompt) for b in active_beams]
    templated_convs = tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    )

    gen_results = generate_k_steps(
        templated_convs,
        lookahead,
        llm,
        sampling_params,
        1,
    )

    for beam, gen_result in zip(active_beams, gen_results, strict=True):
        beam.next_texts = gen_result.next_texts
        beam.stop_reasons = gen_result.stop_reasons
        beam.lookahead_texts = gen_result.lookahead_texts
        beam.completion_tokens += gen_result.completion_tokens
        beam.current_text += beam.next_texts[0]
        beam.history.append(beam.next_texts[0])

        if (
            beam.stop_reasons[0] == "EOS"
            or beam.stop_reasons[0] == "length"
            or beam.next_texts[0] == ""
        ):
            beam.completed = True


def score_and_prune_beams(
    active_beams: List,
    completed_beams: List,
    prm: torch.nn.Module,
    config: torch.nn.Module,
):
    """give step score and select top beam width samples"""

    prompts = [b.prompt for b in active_beams]
    completions = [b.current_text for b in active_beams]
    scores = prm.score(prompts, completions)
    torch.cuda.empty_cache()
    scores = [s if len(s) > 0 else np.array(np.float32(0)) for s in scores]
    agg_scores = [s[-1] for s in scores]

    for beam, score in zip(active_beams, scores, strict=True):
        beam.all_scores = score

    still_active = []
    still_active_scores = []

    for beam, score in zip(active_beams, agg_scores, strict=True):
        if beam.completed:
            completed_beams.append(beam)
        else:
            still_active.append(beam)
            still_active_scores.append(score)

    if len(still_active) == 0:
        return [], completed_beams

    # Beam Pruning
    top_indices = np.argsort(np.array(still_active_scores).flatten())[
        -(config.n // config.beam_width) :
    ]
    for idx, beam in enumerate(still_active):
        if idx not in top_indices:
            beam.pruned = True

    filtered_beams = [b for idx, b in enumerate(still_active) if idx in top_indices]
    return filtered_beams, completed_beams


def run_beam_search(
    prompt: str,
    config: BeamSearchConfig,
    system_prompt: str,
    llm: torch.nn.Module,
    prm: torch.nn.Module,
    progress_bar=True,
):
    active_beams = create_initial_beams(prompt, config)

    completed_beams: List[Beam] = []
    p_bar = (
        tqdm(range(config.num_iterations), desc="Beam Search Depth")
        if progress_bar
        else range(config.num_iterations)
    )
    for i in p_bar:

        active_beams = [b for b in active_beams if not b.pruned]
        active_beams = adjust_active_beams(active_beams, config)

        last_round = i == config.num_iterations - 1
        sampling_params = prepare_iteration_sampling_params(config, last_round)

        generate_for_active_beams(
            active_beams, system_prompt, llm, config, sampling_params, i
        )

        active_beams, completed_beams = score_and_prune_beams(
            active_beams, completed_beams, prm, config
        )

        # * early stop if all beams are purned or enough beams have been completed
        if len(active_beams) == 0 or (
            config.early_stop_criteria is not None
            and len(completed_beams) >= config.early_stop_criteria
        ):
            break

    completed_beams = sorted(
        completed_beams, key=lambda b: b.all_scores[-1], reverse=True
    )
    completed_beams = completed_beams[: config.n]

    torch.cuda.empty_cache()
    if len(completed_beams) == 0:
        Warning("Not Complete!")
        return active_beams
    return completed_beams
