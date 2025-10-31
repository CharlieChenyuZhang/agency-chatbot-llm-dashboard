#!/usr/bin/env python
# coding: utf-8

import os
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sklearn.model_selection
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Local imports
import sys
sys.path.append('src/')
from dataset import TextDataset
from probes import LinearProbeClassification
from train_test_utils import train, test

from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Hugging Face auth and model
# ---------------------------
access_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_ACCESS_TOKEN')
if not access_token:
    raise ValueError("HuggingFace token not found. Please set one of: HF_TOKEN, HUGGINGFACE_TOKEN, HF_ACCESS_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token, padding_side='left')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token)
model.half().cuda()
model.eval()


# ---------------------------
# Config
# ---------------------------
class TrainerConfig:
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Old-logic toggles (mirroring the prior notebook script)
new_prompt_format = True
residual_stream = True
uncertainty = False
logistic = True
augmented = False
remove_last_ai_response = True
include_inst = True
one_hot = True

# Label mapping for age
label_to_id_age = {
    "child": 0,
    "adolescent": 1,
    "adult": 2,
    "older adult": 3,
}

# Data directories (old logic, OpenAI age, set to extracted folders)
data_root = os.path.join("data", "dataset", "original-talktuner-data")
primary_dir = os.path.join(data_root, "openai_age_1")
additional_dirs = [
    os.path.join(data_root, "openai_age_2"),
]


# ---------------------------
# Output/checkpoints
# ---------------------------
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
output_root = os.path.join("output", run_timestamp)
ckpt_root = os.path.join(output_root, "probe_checkpoints", "reading_probe")
os.makedirs(ckpt_root, exist_ok=True)


def main():
    torch_device = "cuda"
    dict_name = "age"

    # Build dataset using old logic API
    dataset = TextDataset(
        primary_dir,
        tokenizer,
        model,
        label_idf="_age_",
        label_to_id=label_to_id_age,
        convert_to_llama2_format=True,
        additional_datas=additional_dirs,
        new_format=new_prompt_format,
        residual_stream=residual_stream,
        if_augmented=augmented,
        remove_last_ai_response=remove_last_ai_response,
        include_inst=include_inst,
        k=1,
        one_hot=False,
        last_tok_pos=-1,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Label distribution: {dict(zip(*np.unique(dataset.labels, return_counts=True)))}")

    # Train/val split (80/20 as in old logic)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        train_size=train_size,
        random_state=12345,
        shuffle=True,
        stratify=dataset.labels,
    )

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=200, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=400, num_workers=1)

    # Loss per old logic (logistic + one_hot -> BCE)
    loss_func = nn.BCELoss() if not uncertainty else None

    accuracy_record = {dict_name: [], f"{dict_name}_final": [], f"{dict_name}_train": []}

    accs, final_accs, train_accs = [], [], []

    for layer_num in tqdm(range(0, 41), desc=f"Training {dict_name} probes"):
        trainer_config = TrainerConfig()
        probe = LinearProbeClassification(
            device=torch_device,
            probe_class=len(label_to_id_age.keys()),
            input_dim=5120,
            logistic=logistic,
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_config)

        best_acc = 0.0
        max_epoch = 50
        verbosity = False
        print("-" * 40 + f"Layer {layer_num}" + "-" * 40)

        for epoch in range(1, max_epoch + 1):
            if epoch == max_epoch:
                verbosity = True

            train_results = train(
                probe,
                torch_device,
                train_loader,
                optimizer,
                epoch,
                loss_func=loss_func,
                verbose_interval=None,
                verbose=verbosity,
                layer_num=layer_num,
                return_raw_outputs=True,
                one_hot=one_hot,
                num_classes=len(label_to_id_age.keys()),
            )

            test_results = test(
                probe,
                torch_device,
                test_loader,
                loss_func=loss_func,
                return_raw_outputs=True,
                verbose=verbosity,
                layer_num=layer_num,
                scheduler=scheduler,
                one_hot=one_hot,
                num_classes=len(label_to_id_age.keys()),
            )

            if test_results[1] > best_acc:
                best_acc = test_results[1]
                torch.save(
                    probe.state_dict(),
                    os.path.join(ckpt_root, f"{dict_name}_probe_at_layer_{layer_num}.pth"),
                )

        # Save final model for this layer
        torch.save(
            probe.state_dict(),
            os.path.join(ckpt_root, f"{dict_name}_probe_at_layer_{layer_num}_final.pth"),
        )

        accs.append(best_acc)
        final_accs.append(test_results[1])
        train_accs.append(train_results[1])

        # Confusion matrix per layer
        cm = confusion_matrix(test_results[3], test_results[2])
        ConfusionMatrixDisplay(cm, display_labels=list(label_to_id_age.keys())).plot()
        plt.title(f"Age - Layer {layer_num}")
        plt.savefig(os.path.join(output_root, f"confusion_matrix_age_layer_{layer_num}.png"))
        plt.close()

        accuracy_record[dict_name].append(accs)
        accuracy_record[f"{dict_name}_final"].append(final_accs)
        accuracy_record[f"{dict_name}_train"].append(train_accs)

        # Persist running results
        with open(os.path.join(output_root, "reading_probe_experiment.pkl"), "wb") as outfile:
            pickle.dump(accuracy_record, outfile)

    # Cleanup
    del dataset, train_dataset, test_dataset, train_loader, test_loader
    torch.cuda.empty_cache()

    # Summary plot for best accuracies across layers
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(accs)), accs, 'b-', label='Best Accuracy')
    plt.title('Age Probe Accuracy (Old Logic)')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "age_probe_accuracy_old_logic.png"))
    plt.close()

    print("\nBest Results:")
    best_layer = int(np.argmax(accs)) if len(accs) else -1
    best_acc = float(max(accs)) if len(accs) else 0.0
    print(f"Age: {best_acc:.3f} at layer {best_layer}")


if __name__ == "__main__":
    main()


