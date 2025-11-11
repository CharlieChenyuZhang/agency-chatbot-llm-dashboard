#!/usr/bin/env python
# coding: utf-8

# # Training Behavioral Trait Probes
# 
# This notebook trains probes to detect behavioral traits: Rigidity, Independence, and Goal Persistence.
# 

# In[ ]:


# Jupyter magic commands removed for script execution
# %load_ext autoreload
# %autoreload 2


# In[ ]:


import os
import sys
sys.path.append('src/')

# Enable tokenizers parallelism for better performance with multiple GPUs
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from losses import edl_mse_loss

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm

from behavioral_dataset import BehavioralTraitDataset, create_behavioral_dataset
from behavioral_traits_config import (
    BEHAVIORAL_TRAIT_LABELS, 
    BEHAVIORAL_DATASET_DIRS,
    BEHAVIORAL_TRAINING_CONFIG
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from probes import ProbeClassification, ProbeClassificationMixScaler, LinearProbeClassification
from train_test_utils import train, test 
import torch.nn as nn

import time
import pickle
import sklearn.model_selection
import numpy as np
import atexit

tic, toc = (time.time, time.time)


# In[ ]:


# Load model and tokenizer
access_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_ACCESS_TOKEN')

if not access_token:
    raise ValueError("HuggingFace token not found. Please set one of these environment variables: HF_TOKEN, HUGGINGFACE_TOKEN, or HF_ACCESS_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token, padding_side='left')
# Select best available device
if torch.cuda.is_available():
    torch_device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch_device = "mps"
else:
    torch_device = "cpu"

# Prefer float16 on GPU/MPS, float32 on CPU
preferred_dtype = torch.float16 if torch_device in ("cuda", "mps") else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",
    token=access_token,
    torch_dtype=preferred_dtype
)
model.to(torch_device)
model.eval()


# In[ ]:


class TrainerConfig:
    # optimization parameters
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1 # only applied on matmul weights

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


# ## Training Configuration
# 

# In[ ]:


# Training configuration
new_prompt_format = True
residual_stream = True
uncertainty = False
logistic = True
augmented = False
remove_last_ai_response = True
include_inst = True
one_hot = True  # Enable one-hot targets for BCE
regression_mode = False  # Set to True for continuous prediction

# Behavioral traits to train
behavioral_traits = ["rigidity", "independence", "goal_persistence"]

accuracy_dict = {}

# Dataset family toggle: choose between 'gpt5' and 'llama2' (can override via env BEHAVIORAL_DATASET_FAMILY)
DATASET_FAMILY = 'gpt5' # or 'gpt5'

# Build selected dataset directories based on the chosen family
if DATASET_FAMILY == 'gpt5':
    SELECTED_BEHAVIORAL_DATASET_DIRS = BEHAVIORAL_DATASET_DIRS
else:
    # Default branch maps gpt5_* directories to llama2_* by string replacement
    SELECTED_BEHAVIORAL_DATASET_DIRS = {
        trait: [p.replace('gpt5_', 'llama2_') for p in paths]
        for trait, paths in BEHAVIORAL_DATASET_DIRS.items()
    }

# Derive a dataset tag (e.g., "gpt5" or "llama2") once for this run from the selected directories
_primary_dataset_dir = SELECTED_BEHAVIORAL_DATASET_DIRS[behavioral_traits[0]][0]
_primary_dataset_folder = os.path.basename(os.path.normpath(_primary_dataset_dir))
dataset_tag = _primary_dataset_folder.split('_')[0] if '_' in _primary_dataset_folder else _primary_dataset_folder

# Timestamped output directory
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
output_root = os.path.join("output", run_timestamp)
checkpoint_dir = os.path.join(output_root, "probe_checkpoints", "behavioral_probes")
os.makedirs(checkpoint_dir, exist_ok=True)

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Mirror all prints (stdout/stderr) to a log file in the output folder
_orig_stdout, _orig_stderr = sys.stdout, sys.stderr
_log_path = os.path.join(output_root, "train_behavioral_traits.log")
_log_file = open(_log_path, "a", buffering=1)
sys.stdout = _TeeIO(sys.stdout, _log_file)
sys.stderr = _TeeIO(sys.stderr, _log_file)

def _restore_streams_and_close():
    try:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
    except Exception:
        pass
    try:
        _log_file.flush()
        _log_file.close()
    except Exception:
        pass

atexit.register(_restore_streams_and_close)
print(f"[logging] Mirroring stdout/stderr to {_log_path}")

# ## Training Loop for Behavioral Traits
# 

# In[ ]:


for trait_type in behavioral_traits:
    print(f"\n{'='*60}")
    print(f"Training {trait_type.upper()} probe")
    print(f"{'='*60}")
    
    # Get directories for this trait
    directories = SELECTED_BEHAVIORAL_DATASET_DIRS[trait_type]
    
    # Create dataset
    dataset = create_behavioral_dataset(
        trait_type=trait_type,
        directory=directories[0],  # Use first directory as primary
        tokenizer=tokenizer,
        model=model,
        convert_to_llama2_format=True,
        additional_datas=directories[1:] if len(directories) > 1 else None,
        new_format=new_prompt_format,
        residual_stream=residual_stream,
        if_augmented=augmented,
        remove_last_ai_response=remove_last_ai_response,
        include_inst=include_inst,
        k=1,
        one_hot=False,  # keep raw index labels; one-hot will be applied in train/test
        regression_mode=regression_mode
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Label distribution: {dict(zip(*np.unique(dataset.labels, return_counts=True)))}")
    
    # Train-test split
    train_size = int(BEHAVIORAL_TRAINING_CONFIG['train_split'] * len(dataset))
    test_size = len(dataset) - train_size

    # Build a 1D stratification vector from labels (handles one-hot or index labels)
    if not regression_mode:
        labels_np = np.array(dataset.labels)
        if labels_np.ndim >= 2:
            stratify_labels = labels_np.argmax(axis=-1)
        else:
            stratify_labels = labels_np
    else:
        stratify_labels = None

    train_idx, val_idx = sklearn.model_selection.train_test_split(
        list(range(len(dataset))), 
        test_size=test_size,
        train_size=train_size,
        random_state=BEHAVIORAL_TRAINING_CONFIG['random_state'],
        shuffle=True,
        stratify=stratify_labels
    )

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        pin_memory=True, 
        batch_size=BEHAVIORAL_TRAINING_CONFIG['batch_size'], 
        num_workers=1
    )
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=BEHAVIORAL_TRAINING_CONFIG['test_batch_size'], 
        num_workers=1
    )

    # Loss function
    if uncertainty:
        loss_func = edl_mse_loss
    elif regression_mode:
        loss_func = nn.MSELoss()  # Use MSE for regression
    else:
        loss_func = nn.BCELoss()  # Use BCE for one-hot multi-class (with sigmoid outputs)

    # Initialize accuracy tracking
    accuracy_dict[trait_type] = []
    accuracy_dict[trait_type + "_final"] = []
    accuracy_dict[trait_type + "_train"] = []
    
    accs = []
    final_accs = []
    train_accs = []
    
    # Train probes for each layer
    for i in tqdm(range(0, 41), desc=f"Training {trait_type} probes"):
        trainer_config = TrainerConfig()
        
        # Create probe
        num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type]) if not regression_mode else 1
        probe = LinearProbeClassification(
            probe_class=num_classes, 
            device="cuda", 
            input_dim=5120,
            logistic=logistic
        )
        
        optimizer, scheduler = probe.configure_optimizers(trainer_config)
        best_acc = 0
        max_epoch = BEHAVIORAL_TRAINING_CONFIG['max_epochs']
        verbosity = False
        layer_num = i
        
        print(f"\n{'-' * 40} Layer {layer_num} {'-' * 40}")
        
        # Track per-epoch losses for this layer
        layer_train_losses = []
        layer_test_losses = []
        
        for epoch in range(1, max_epoch + 1):
            if epoch == max_epoch:
                verbosity = True
            
            # Training
            if uncertainty:
                train_results = train(
                    probe, torch_device, train_loader, optimizer, 
                    epoch, loss_func=loss_func, verbose_interval=None,
                    verbose=verbosity, layer_num=layer_num, 
                    return_raw_outputs=True, epoch_num=epoch, 
                    num_classes=num_classes
                )
                test_results = test(
                    probe, torch_device, test_loader, loss_func=loss_func, 
                    return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                    scheduler=scheduler, epoch_num=epoch, 
                    num_classes=num_classes
                )
            else:
                train_results = train(
                    probe, torch_device, train_loader, optimizer, 
                    epoch, loss_func=loss_func, verbose_interval=None,
                    verbose=verbosity, layer_num=layer_num,
                    return_raw_outputs=True,
                    one_hot=one_hot, num_classes=num_classes
                )
                test_results = test(
                    probe, torch_device, test_loader, loss_func=loss_func, 
                    return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                    scheduler=scheduler,
                    one_hot=one_hot, num_classes=num_classes
                )

            # Record per-epoch loss
            layer_train_losses.append(train_results[0])
            layer_test_losses.append(test_results[0])

            # Save best model
            if test_results[1] > best_acc:
                best_acc = test_results[1]
                torch.save(
                    probe.state_dict(), 
                    os.path.join(checkpoint_dir, f"{trait_type}_{dataset_tag}_probe_at_layer_{layer_num}.pth")
                )
        
        # Save final model
        torch.save(
            probe.state_dict(), 
            os.path.join(checkpoint_dir, f"{trait_type}_{dataset_tag}_probe_at_layer_{layer_num}_final.pth")
        )
        
        accs.append(best_acc)
        final_accs.append(test_results[1])
        train_accs.append(train_results[1])
        
        # Plot per-epoch loss curves for this layer
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(layer_train_losses)+1), layer_train_losses, label='Train Loss')
        plt.plot(range(1, len(layer_test_losses)+1), layer_test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{trait_type.capitalize()} - Layer {layer_num} Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_root, f"loss_curve_{trait_type}_{dataset_tag}_layer_{layer_num}.png"))
        plt.close()
        
        # Plot confusion matrix
        if not regression_mode:
            cm = confusion_matrix(test_results[3], test_results[2])
            cm_display = ConfusionMatrixDisplay(
                cm, 
                display_labels=list(BEHAVIORAL_TRAIT_LABELS[trait_type].keys())
            ).plot()
            plt.title(f"{trait_type.capitalize()} - Layer {layer_num}")
            plt.savefig(os.path.join(output_root, f"confusion_matrix_{trait_type}_{dataset_tag}_layer_{layer_num}.png"))
            plt.close()

        # Update accuracy dict
        accuracy_dict[trait_type].append(accs)
        accuracy_dict[trait_type + "_final"].append(final_accs)
        accuracy_dict[trait_type + "_train"].append(train_accs)
        
        # Save intermediate results
        with open(os.path.join(output_root, f"probe_checkpoints/behavioral_probes_experiment_{dataset_tag}.pkl"), "wb") as outfile:
            pickle.dump(accuracy_dict, outfile)
    
    # Clean up
    del dataset, train_dataset, test_dataset, train_loader, test_loader
    torch.cuda.empty_cache()

print("\nTraining completed for all behavioral traits!")


# ## Results Analysis
# 

# In[ ]:


# Plot results for each trait
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, trait_type in enumerate(behavioral_traits):
    if trait_type in accuracy_dict:
        accs = accuracy_dict[trait_type][-1]  # Get the last (complete) results
        axes[i].plot(range(len(accs)), accs, 'b-', label='Best Accuracy')
        axes[i].set_title(f'{trait_type.capitalize()} Probe Accuracy')
        axes[i].set_xlabel('Layer')
        axes[i].set_ylabel('Accuracy')
        axes[i].grid(True)
        axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_root, f"behavioral_traits_accuracy_plots_{dataset_tag}.png"))
plt.close()

# Print best results for each trait
print("\nBest Results:")
for trait_type in behavioral_traits:
    if trait_type in accuracy_dict:
        accs = accuracy_dict[trait_type][-1]
        best_layer = np.argmax(accs)
        best_acc = max(accs)
        print(f"{trait_type.capitalize()}: {best_acc:.3f} at layer {best_layer}")


# 
