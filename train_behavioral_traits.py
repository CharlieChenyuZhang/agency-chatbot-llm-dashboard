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

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

tic, toc = (time.time, time.time)


# In[ ]:


# Load model and tokenizer
access_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_ACCESS_TOKEN')

if not access_token:
    raise ValueError("HuggingFace token not found. Please set one of these environment variables: HF_TOKEN, HUGGINGFACE_TOKEN, or HF_ACCESS_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token, padding_side='left')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token=access_token)
model.half().cuda()
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
one_hot = False  # Set to False for behavioral traits
regression_mode = False  # Set to True for continuous prediction

# Behavioral traits to train
behavioral_traits = ["rigidity", "independence", "goal_persistence"]

accuracy_dict = {}
torch_device = "cuda"


# ## Training Loop for Behavioral Traits
# 

# In[ ]:


for trait_type in behavioral_traits:
    print(f"\n{'='*60}")
    print(f"Training {trait_type.upper()} probe")
    print(f"{'='*60}")
    
    # Get directories for this trait
    directories = BEHAVIORAL_DATASET_DIRS[trait_type]
    
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
        one_hot=one_hot,
        regression_mode=regression_mode
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Label distribution: {dict(zip(*np.unique(dataset.labels, return_counts=True)))}")
    
    # Train-test split
    train_size = int(BEHAVIORAL_TRAINING_CONFIG['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        list(range(len(dataset))), 
        test_size=test_size,
        train_size=train_size,
        random_state=BEHAVIORAL_TRAINING_CONFIG['random_state'],
        shuffle=True,
        stratify=dataset.labels if not regression_mode else None
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
        loss_func = nn.CrossEntropyLoss()  # Use CrossEntropy for classification

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

            # Save best model
            if test_results[1] > best_acc:
                best_acc = test_results[1]
                torch.save(
                    probe.state_dict(), 
                    f"probe_checkpoints/behavioral_probes/{trait_type}_probe_at_layer_{layer_num}.pth"
                )
        
        # Save final model
        torch.save(
            probe.state_dict(), 
            f"probe_checkpoints/behavioral_probes/{trait_type}_probe_at_layer_{layer_num}_final.pth"
        )
        
        accs.append(best_acc)
        final_accs.append(test_results[1])
        train_accs.append(train_results[1])
        
        # Plot confusion matrix
        if not regression_mode:
            cm = confusion_matrix(test_results[3], test_results[2])
            cm_display = ConfusionMatrixDisplay(
                cm, 
                display_labels=list(BEHAVIORAL_TRAIT_LABELS[trait_type].keys())
            ).plot()
            plt.title(f"{trait_type.capitalize()} - Layer {layer_num}")
            plt.savefig(f"confusion_matrix_{trait_type}_layer_{layer_num}.png")
            plt.close()

        # Update accuracy dict
        accuracy_dict[trait_type].append(accs)
        accuracy_dict[trait_type + "_final"].append(final_accs)
        accuracy_dict[trait_type + "_train"].append(train_accs)
        
        # Save intermediate results
        with open("probe_checkpoints/behavioral_probes_experiment.pkl", "wb") as outfile:
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
plt.savefig("behavioral_traits_accuracy_plots.png")
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
