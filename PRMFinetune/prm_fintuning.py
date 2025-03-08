"""
Parts of this code are referenced from the openr project.
For more details, please refer to https://github.com/openreasoner/openr.
"""

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


good_token = "+"
bad_token = "-"
step_tag = "\n\n\n\n\n"
step_tag2 = "\n\n"
model_path = "microsoft/Phi-4-mini-instruct"
DATA_PATH = "/home/MATH_APS_clean.json"
per_device_train_batch_size = 2
per_device_eval_batch_size = 8
total_batch_size = 16
learning_rate = 1e-4
BATCH_SIZE = total_batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // per_device_train_batch_size
fp = f"bs_{total_batch_size}_lr_{learning_rate}"
output_path = f"./prm_results/{fp}"


def preprocess_function(example):
    input = f"{example['question']} {example['process']}"
    tokenized_inputs = tokenizer(
        input,
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]

    length = len(tokenized_inputs["input_ids"])
    indices = find_all_indices(tokenized_inputs["input_ids"], step_tag_id)

    if len(indices) != len(example["label"]):
        example["label"] = example["label"][: len(indices)]

    assert len(indices) == len(example["label"])

    tokenized_inputs["labels"] = [-100] * length
    for i in range(len(indices)):
        if example["label"][i] == "+" or example["label"][i] == 1:
            tokenized_inputs["labels"][indices[i]] = candidate_tokens[0]
        elif example["label"][i] == "-" or example["label"][i] == 0:
            tokenized_inputs["labels"][indices[i]] = candidate_tokens[1]
        else:
            raise ValueError("label is wrong")
        tokenized_inputs["attention_mask"][indices[i]] = 0
    return tokenized_inputs


class SaveBeforeEvaluateCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        trainer.save_model(output_dir="./saved_model_before_eval")


def compute_metrics(eval_pred):
    pre, labels = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result = {
        "auc": auc,
        "ll": ll,
        "acc": acc,
    }
    return result


def preprocess_logits_for_metrics(logits, labels):
    # return logits,labels
    labels_index = torch.argwhere(
        torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1])
    )
    gold = torch.where(
        labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1
    )
    logits = logits[labels_index[:, 0], labels_index[:, 1]][
        :, [candidate_tokens[1], candidate_tokens[0]]
    ]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,  # Enables 8-bit quantization
        device_map="auto",  # Automatically assigns the model to available GPUs/CPUs
        torch_dtype="auto",  # Mixed precision for faster inference
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=24,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["qkv_proj"],  # Apply LoRA to specific layers
    )
    model = get_peft_model(model, lora_config)
    dataset = load_dataset("json", data_files=DATA_PATH)
    dataset["train"] = dataset["train"]
    dataset_train_valtest = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset_valtest = dataset_train_valtest["test"].train_test_split(
        test_size=0.5, seed=42
    )
    dataset = DatasetDict(
        {
            "train": dataset_train_valtest["train"],
            "valid": dataset_valtest["train"],
            "test": dataset_valtest["test"],
        }
    )

    tokenized_datasets = dataset.map(preprocess_function)
    tokenized_datasets["train"] = tokenized_datasets["train"].remove_columns(
        ["question", "process", "label"]
    )
    tokenized_datasets["valid"] = tokenized_datasets["valid"].remove_columns(
        ["question", "process", "label"]
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="no",
        eval_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1,
        warmup_ratio=0.1,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        callbacks=[SaveBeforeEvaluateCallback()],
    )

    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_prm_lora_4bit")
    tokenizer.save_pretrained("./fine_tuned_prm_lora_4bit")
