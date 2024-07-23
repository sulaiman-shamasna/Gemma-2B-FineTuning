import torch
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding)

import bitsandbytes as bnb

import evaluate
import numpy as np

import random

dataset_imdb = load_dataset("imdb")
print(type(dataset_imdb))

reduction_rate    = 0.1
num_train_to_keep = int(reduction_rate * dataset_imdb["train"].num_rows)
num_test_to_keep  = int(reduction_rate * dataset_imdb["test"].num_rows)

def select_random_indices(dataset, num_to_keep):
    indices = list(range(dataset.num_rows))
    random.shuffle(indices)
    return indices[:num_to_keep]

train_indices = select_random_indices(dataset_imdb["train"], num_train_to_keep)
test_indices  = select_random_indices(dataset_imdb["test"], num_test_to_keep)

dataset_imdb  = DatasetDict({
    "train": dataset_imdb["train"].select(train_indices),
    "test": dataset_imdb["test"].select(test_indices),
})

model_id  = "gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f' Vocab size of the model {model_id}: {len(tokenizer.get_vocab())}')
#Vocab size of the model google/gemma-2b-it: 256000


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_imdb = dataset_imdb.map(preprocess_function, batched=True)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


import sklearn
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert probabilities to predicted labels
    return metric.compute(predictions=predictions, references=labels)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enables 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for potentially higher accuracy (optional)
    bnb_4bit_quant_type="nf4",  # Quantization type (specifics depend on hardware and library)
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for improved efficiency (optional)
)


import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of available GPUs
print(torch.cuda.get_device_name(0))  # Should return the name of the GPUpython ge  


model = AutoModelForSequenceClassification.from_pretrained(
    model_id,  # "google/gemma-2b-it"
    num_labels=2,  # Number of output labels (2 for binary sentiment classification)
    id2label=id2label,  # {0: "NEGATIVE", 1: "POSITIVE"} 
    label2id=label2id,  # {"NEGATIVE": 0, "POSITIVE": 1}
    quantization_config=bnb_config,  # configuration for quantization 
    device_map={"": 0}  # Optional dictionary specifying device mapping (single GPU with index 0 here)
)