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
combine_layers = False  # If True, concatenate all layers into one feature vector per sample

# Probe type configuration
# Options: "reading", "control", or "both"
# - "reading": Train reading probes (control_probe=False, adds suffix for detection)
# - "control": Train control probes (control_probe=True, no suffix, better for steering)
# - "both": Train both types sequentially (recommended)
PROBE_TYPE = "both"  # Default to training both probe types

if PROBE_TYPE not in ["reading", "control", "both"]:
    raise ValueError(f"PROBE_TYPE must be 'reading', 'control', or 'both', got: {PROBE_TYPE}")

# Determine which probe types to train
if PROBE_TYPE == "both":
    probe_types_to_train = ["reading", "control"]
else:
    probe_types_to_train = [PROBE_TYPE]

print(f"\n{'='*60}")
print(f"Probe Type Configuration: {PROBE_TYPE}")
print(f"Will train: {probe_types_to_train}")
print(f"{'='*60}\n")

# Behavioral traits to train
behavioral_traits = ["rigidity", "independence", "goal_persistence"]

accuracy_dict = {}

# Dataset family toggle: choose between 'gpt5' and 'llama2' (can override via env BEHAVIORAL_DATASET_FAMILY)
DATASET_FAMILY = 'gpt5' # or 'gpt5'

# Use training data from final_trainingdata_gpt5 directory
BASE_DATASET_DIR = "data/dataset/final_trainingdata_gpt5_flatten"

# Build selected dataset directories based on the chosen family
if DATASET_FAMILY == 'gpt5':
    # Override to use final_trainingdata_gpt5 directory
    SELECTED_BEHAVIORAL_DATASET_DIRS = {
        "rigidity": [os.path.join(BASE_DATASET_DIR, "gpt5_rigidity_1/")],
        "independence": [os.path.join(BASE_DATASET_DIR, "gpt5_independence_1/")],
        "goal_persistence": [os.path.join(BASE_DATASET_DIR, "gpt5_goal_persistence_1/")]
    }
else:
    # Default branch maps gpt5_* directories to llama2_* by string replacement
    SELECTED_BEHAVIORAL_DATASET_DIRS = {
        trait: [p.replace('gpt5_', 'llama2_') for p in paths]
        for trait, paths in BEHAVIORAL_DATASET_DIRS.items()
    }

# Filter behavioral traits to only include those with existing directories
behavioral_traits = [trait for trait in behavioral_traits 
                     if trait in SELECTED_BEHAVIORAL_DATASET_DIRS 
                     and os.path.exists(SELECTED_BEHAVIORAL_DATASET_DIRS[trait][0])]

if not behavioral_traits:
    raise ValueError(f"No valid dataset directories found in {BASE_DATASET_DIR}")

print(f"Training probes for traits: {behavioral_traits}")
print(f"Found {len(behavioral_traits)} trait(s) with valid dataset directories")
for trait in behavioral_traits:
    print(f"  - {trait}: {SELECTED_BEHAVIORAL_DATASET_DIRS[trait][0]}")

# Derive a dataset tag (e.g., "gpt5" or "llama2") once for this run from the selected directories
_primary_dataset_dir = SELECTED_BEHAVIORAL_DATASET_DIRS[behavioral_traits[0]][0]
_primary_dataset_folder = os.path.basename(os.path.normpath(_primary_dataset_dir))
dataset_tag = _primary_dataset_folder.split('_')[0] if '_' in _primary_dataset_folder else _primary_dataset_folder

# Timestamped output directory
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
output_root = os.path.join("output", run_timestamp)

# Create checkpoint directories based on probe type
if PROBE_TYPE == "both":
    reading_checkpoint_dir = os.path.join(output_root, "probe_checkpoints", "reading_probe")
    control_checkpoint_dir = os.path.join(output_root, "probe_checkpoints", "control_probe")
    os.makedirs(reading_checkpoint_dir, exist_ok=True)
    os.makedirs(control_checkpoint_dir, exist_ok=True)
else:
    if PROBE_TYPE == "reading":
        checkpoint_dir = os.path.join(output_root, "probe_checkpoints", "reading_probe")
    else:  # control
        checkpoint_dir = os.path.join(output_root, "probe_checkpoints", "control_probe")
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

# Outer loop: train each probe type
for probe_type_name in probe_types_to_train:
    is_control_probe = (probe_type_name == "control")
    
    # Set checkpoint directory for this probe type
    if PROBE_TYPE == "both":
        current_checkpoint_dir = reading_checkpoint_dir if probe_type_name == "reading" else control_checkpoint_dir
    else:
        current_checkpoint_dir = checkpoint_dir
    
    print(f"\n{'='*80}")
    print(f"Training {probe_type_name.upper()} probes (control_probe={is_control_probe})")
    print(f"Checkpoint directory: {current_checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Reset accuracy dict for this probe type
    accuracy_dict = {}
    
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
        regression_mode=regression_mode,
        control_probe=is_control_probe  # Set based on probe type
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

    # Extract all features and labels into tensors for GPU-efficient training
    print("Extracting features to GPU tensors...")
    # Stack all features: [N, 41, 5120] where N is number of samples
    all_features = torch.stack([dataset.acts[i] for i in range(len(dataset))])
    all_labels = torch.tensor(dataset.labels, dtype=torch.long if not regression_mode else torch.float32)
    
    # Split into train/test
    train_features = all_features[train_idx]  # [N_train, 41, 5120]
    test_features = all_features[val_idx]      # [N_test, 41, 5120]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[val_idx]
    
    # Move to GPU with non_blocking for better performance
    train_features = train_features.to(torch_device, non_blocking=True)
    test_features = test_features.to(torch_device, non_blocking=True)
    train_labels = train_labels.to(torch_device, non_blocking=True)
    test_labels = test_labels.to(torch_device, non_blocking=True)
    
    # Convert labels to one-hot if needed
    if one_hot and not regression_mode:
        num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type])
        train_labels = F.one_hot(train_labels.long(), num_classes=num_classes).float()
        test_labels = F.one_hot(test_labels.long(), num_classes=num_classes).float()
    
    print(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")
    print(f"Train labels shape: {train_labels.shape}, Test labels shape: {test_labels.shape}")

    # Loss function
    if uncertainty:
        loss_func = edl_mse_loss
    elif regression_mode:
        loss_func = nn.MSELoss()  # Use MSE for regression
    else:
        loss_func = nn.BCELoss() if one_hot else nn.CrossEntropyLoss()  # Use BCE for one-hot, CE for index labels

    # Initialize accuracy tracking
    accuracy_dict[trait_type] = []
    accuracy_dict[trait_type + "_final"] = []
    accuracy_dict[trait_type + "_train"] = []
    
    accs = []
    final_accs = []
    train_accs = []
    
    # Train probes - either per layer or on combined layers
    if combine_layers:
        # Combine all layers into one feature matrix: [N, 41*5120]
        print("Combining features from all layers into one feature matrix...")
        train_X = train_features.reshape(train_features.shape[0], -1)  # [N_train, 41*5120]
        test_X = test_features.reshape(test_features.shape[0], -1)    # [N_test, 41*5120]
        
        # Train single probe on combined features
        trainer_config = TrainerConfig()
        num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type]) if not regression_mode else 1
        combined_input_dim = 41 * 5120  # Combined feature dimension
        
        probe = LinearProbeClassification(
            probe_class=num_classes, 
            device=torch_device, 
            input_dim=combined_input_dim,
            logistic=logistic
        )
        probe = probe.to(torch_device, non_blocking=True)
        
        optimizer, scheduler = probe.configure_optimizers(trainer_config)
        best_acc = 0
        max_epoch = BEHAVIORAL_TRAINING_CONFIG['max_epochs']
        
        print(f"\n{'-' * 40} Combined Layers (41*5120={combined_input_dim}) {'-' * 40}")
        
        layer_train_losses = []
        layer_test_losses = []
        
        for epoch in range(1, max_epoch + 1):
            verbosity = (epoch == max_epoch)
            
            # Training on full tensor
            probe.train()
            optimizer.zero_grad()
            
            logits, _ = probe(train_X, None)
            
            if regression_mode:
                loss = loss_func(logits.squeeze(), train_labels)
            elif one_hot:
                loss = loss_func(logits, train_labels)
            else:
                loss = loss_func(logits, train_labels.long())
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                if regression_mode:
                    train_pred = logits.squeeze()
                    train_acc = 1.0 - (torch.mean((train_pred - train_labels) ** 2) / torch.var(train_labels)).item()
                else:
                    train_pred = torch.argmax(logits, dim=1)
                    train_target = torch.argmax(train_labels, dim=1) if one_hot else train_labels.long()
                    train_acc = (train_pred == train_target).float().mean().item()
            
            train_loss = loss.item()
            layer_train_losses.append(train_loss)
            
            # Testing
            probe.eval()
            with torch.no_grad():
                test_logits, _ = probe(test_X, None)
                
                if regression_mode:
                    test_loss = loss_func(test_logits.squeeze(), test_labels).item()
                    test_pred = test_logits.squeeze()
                    test_acc = 1.0 - (torch.mean((test_pred - test_labels) ** 2) / torch.var(test_labels)).item()
                elif one_hot:
                    test_loss = loss_func(test_logits, test_labels).item()
                    test_pred = torch.argmax(test_logits, dim=1)
                    test_target = torch.argmax(test_labels, dim=1)
                    test_acc = (test_pred == test_target).float().mean().item()
                else:
                    test_loss = loss_func(test_logits, test_labels.long()).item()
                    test_pred = torch.argmax(test_logits, dim=1)
                    test_target = test_labels.long()
                    test_acc = (test_pred == test_target).float().mean().item()
            
            layer_test_losses.append(test_loss)
            
            if scheduler:
                scheduler.step(test_loss)
            
            if verbosity:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(
                        probe.state_dict(), 
                        os.path.join(current_checkpoint_dir, f"{trait_type}_{dataset_tag}_probe_combined_layers.pth")
                    )
        
        # Store results
        accs.append(best_acc)
        final_accs.append(test_acc)
        train_accs.append(train_acc)
        
        # Plot loss curves
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(layer_train_losses)+1), layer_train_losses, label='Train Loss')
        plt.plot(range(1, len(layer_test_losses)+1), layer_test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{trait_type.capitalize()} - Combined Layers Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_root, f"loss_curve_{trait_type}_{dataset_tag}_{probe_type_name}_combined_layers.png"))
        plt.close()
        
        # Plot confusion matrix
        if not regression_mode:
            test_target_np = test_target.cpu().numpy()
            test_pred_np = test_pred.cpu().numpy()
            
            # Ensure labels are 1D integer arrays
            if test_target_np.ndim > 1:
                test_target_np = np.argmax(test_target_np, axis=-1)
            if test_pred_np.ndim > 1:
                test_pred_np = np.argmax(test_pred_np, axis=-1)
            test_target_np = test_target_np.astype(int)
            test_pred_np = test_pred_np.astype(int)
            
            # Get unique labels present in the data (both target and predictions)
            unique_labels = sorted(set(np.concatenate([test_target_np, test_pred_np])))
            
            # Map label indices to their string keys from config
            label_to_id = BEHAVIORAL_TRAIT_LABELS[trait_type]
            id_to_label = {v: k for k, v in label_to_id.items()}
            
            # Get display labels only for labels that exist in the data
            display_labels = [id_to_label.get(label_idx, str(label_idx)) for label_idx in unique_labels]
            
            # Compute confusion matrix with explicit labels to ensure correct shape
            cm = confusion_matrix(test_target_np, test_pred_np, labels=unique_labels)
            
            cm_display = ConfusionMatrixDisplay(
                cm, 
                display_labels=display_labels
            ).plot()
            plt.title(f"{trait_type.capitalize()} - Combined Layers")
            plt.savefig(os.path.join(output_root, f"confusion_matrix_{trait_type}_{dataset_tag}_{probe_type_name}_combined_layers.png"))
            plt.close()
        
        # Update accuracy dict
        accuracy_dict[trait_type] = accs
        accuracy_dict[trait_type + "_final"] = final_accs
        accuracy_dict[trait_type + "_train"] = train_accs
        
        # Save intermediate results
        results_file = os.path.join(output_root, f"probe_checkpoints/{probe_type_name}_probe_experiment_{dataset_tag}.pkl")
        with open(results_file, "wb") as outfile:
            pickle.dump(accuracy_dict, outfile)
    
    else:
        # Train probes for each layer separately
        for i in tqdm(range(0, 41), desc=f"Training {trait_type} probes"):
            trainer_config = TrainerConfig()
            
            # Create probe - ensure it's on the same device
            num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type]) if not regression_mode else 1
            probe = LinearProbeClassification(
                probe_class=num_classes, 
                device=torch_device, 
                input_dim=5120,
                logistic=logistic
            )
            # Ensure probe is on the correct device
            probe = probe.to(torch_device, non_blocking=True)
            
            optimizer, scheduler = probe.configure_optimizers(trainer_config)
            best_acc = 0
            max_epoch = BEHAVIORAL_TRAINING_CONFIG['max_epochs']
            verbosity = False
            layer_num = i
            
            print(f"\n{'-' * 40} Layer {layer_num} {'-' * 40}")
            
            # Extract features for this layer: [N, 5120]
            train_X = train_features[:, layer_num, :]  # [N_train, 5120]
            test_X = test_features[:, layer_num, :]    # [N_test, 5120]
            
            # Track per-epoch losses for this layer
            layer_train_losses = []
            layer_test_losses = []
            
            for epoch in range(1, max_epoch + 1):
                if epoch == max_epoch:
                    verbosity = True
                
                # Training on full tensor
                probe.train()
                optimizer.zero_grad()
                
                # Forward pass on entire training set
                logits, _ = probe(train_X, None)
                
                # Compute loss
                if regression_mode:
                    loss = loss_func(logits.squeeze(), train_labels)
                elif one_hot:
                    loss = loss_func(logits, train_labels)
                else:
                    loss = loss_func(logits, train_labels.long())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Compute training accuracy
                with torch.no_grad():
                    if regression_mode:
                        train_pred = logits.squeeze()
                        train_acc = 1.0 - (torch.mean((train_pred - train_labels) ** 2) / torch.var(train_labels)).item()
                    else:
                        train_pred = torch.argmax(logits, dim=1)
                        if one_hot:
                            train_target = torch.argmax(train_labels, dim=1)
                        else:
                            train_target = train_labels.long()
                        train_acc = (train_pred == train_target).float().mean().item()
                
                train_loss = loss.item()
                layer_train_losses.append(train_loss)
                
                # Testing on full tensor
                probe.eval()
                with torch.no_grad():
                    test_logits, _ = probe(test_X, None)
                    
                    # Compute test loss
                    if regression_mode:
                        test_loss = loss_func(test_logits.squeeze(), test_labels).item()
                    elif one_hot:
                        test_loss = loss_func(test_logits, test_labels).item()
                    else:
                        test_loss = loss_func(test_logits, test_labels.long()).item()
                    
                    # Compute test accuracy
                    if regression_mode:
                        test_pred = test_logits.squeeze()
                        test_acc = 1.0 - (torch.mean((test_pred - test_labels) ** 2) / torch.var(test_labels)).item()
                    else:
                        test_pred = torch.argmax(test_logits, dim=1)
                        if one_hot:
                            test_target = torch.argmax(test_labels, dim=1)
                        else:
                            test_target = test_labels.long()
                        test_acc = (test_pred == test_target).float().mean().item()
                
                layer_test_losses.append(test_loss)
                
                # Update scheduler
                if scheduler:
                    scheduler.step(test_loss)
                
                if verbosity:
                    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                
                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(
                        probe.state_dict(), 
                        os.path.join(current_checkpoint_dir, f"{trait_type}_{dataset_tag}_probe_at_layer_{layer_num}.pth")
                    )
                
                # Store results for final epoch
                if epoch == max_epoch:
                    train_results = (train_loss, train_acc, train_pred.cpu().numpy(), 
                                   (train_target if not regression_mode else train_labels).cpu().numpy())
                    test_results = (test_loss, test_acc, test_pred.cpu().numpy(), 
                                   (test_target if not regression_mode else test_labels).cpu().numpy())
            
            # Save final model
            torch.save(
                probe.state_dict(), 
                os.path.join(current_checkpoint_dir, f"{trait_type}_{dataset_tag}_probe_at_layer_{layer_num}_final.pth")
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
            plt.savefig(os.path.join(output_root, f"loss_curve_{trait_type}_{dataset_tag}_{probe_type_name}_layer_{layer_num}.png"))
            plt.close()
            
            # Plot confusion matrix
            if not regression_mode:
                test_target_np = test_results[3]
                test_pred_np = test_results[2]
                
                # Ensure labels are 1D integer arrays
                if isinstance(test_target_np, torch.Tensor):
                    test_target_np = test_target_np.cpu().numpy()
                if isinstance(test_pred_np, torch.Tensor):
                    test_pred_np = test_pred_np.cpu().numpy()
                
                if test_target_np.ndim > 1:
                    test_target_np = np.argmax(test_target_np, axis=-1)
                if test_pred_np.ndim > 1:
                    test_pred_np = np.argmax(test_pred_np, axis=-1)
                test_target_np = test_target_np.astype(int)
                test_pred_np = test_pred_np.astype(int)
                
                # Get unique labels present in the data (both target and predictions)
                unique_labels = sorted(set(np.concatenate([test_target_np, test_pred_np])))
                
                # Map label indices to their string keys from config
                label_to_id = BEHAVIORAL_TRAIT_LABELS[trait_type]
                id_to_label = {v: k for k, v in label_to_id.items()}
                
                # Get display labels only for labels that exist in the data
                display_labels = [id_to_label.get(label_idx, str(label_idx)) for label_idx in unique_labels]
                
                # Compute confusion matrix with explicit labels to ensure correct shape
                cm = confusion_matrix(test_target_np, test_pred_np, labels=unique_labels)
                
                cm_display = ConfusionMatrixDisplay(
                    cm, 
                    display_labels=display_labels
                ).plot()
                plt.title(f"{trait_type.capitalize()} - Layer {layer_num}")
                plt.savefig(os.path.join(output_root, f"confusion_matrix_{trait_type}_{dataset_tag}_{probe_type_name}_layer_{layer_num}.png"))
                plt.close()
        
        # Update accuracy dict after all layers are processed
        accuracy_dict[trait_type] = accs
        accuracy_dict[trait_type + "_final"] = final_accs
        accuracy_dict[trait_type + "_train"] = train_accs
        
        # Save intermediate results
        results_file = os.path.join(output_root, f"probe_checkpoints/{probe_type_name}_probe_experiment_{dataset_tag}.pkl")
        with open(results_file, "wb") as outfile:
            pickle.dump(accuracy_dict, outfile)
    
        # Clean up
        del dataset, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels
        if torch_device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"\n✓ Completed training {probe_type_name} probes for {trait_type}")
    
    print(f"\n{'='*80}")
    print(f"✓ Completed training all traits for {probe_type_name} probes")
    print(f"{'='*80}")
    
    # ## Results Analysis for this probe type
    # Plot results for each trait (using this probe type's accuracy_dict)
    num_traits = len(behavioral_traits)
    if num_traits > 0:
        fig, axes = plt.subplots(1, num_traits, figsize=(5 * num_traits, 5))
        # Handle case where there's only one trait (axes won't be iterable)
        if num_traits == 1:
            axes = [axes]
        
        for i, trait_type in enumerate(behavioral_traits):
            if trait_type in accuracy_dict:
                accs = accuracy_dict[trait_type]
                # Handle both cases: direct list (combine_layers=False) or nested list (combine_layers=True)
                if isinstance(accs, list) and len(accs) > 0 and isinstance(accs[0], list):
                    accs = accs[-1]  # Get the last (complete) results from nested list
                # Now accs should be a list of floats
                if isinstance(accs, list) and len(accs) > 0:
                    axes[i].plot(range(len(accs)), accs, 'b-', label='Best Accuracy')
                    axes[i].set_title(f'{trait_type.capitalize()} Probe Accuracy ({probe_type_name})')
                    axes[i].set_xlabel('Layer' if len(accs) > 1 else 'Combined Layers')
                    axes[i].set_ylabel('Accuracy')
                    axes[i].grid(True)
                    axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_root, f"behavioral_traits_accuracy_plots_{dataset_tag}_{probe_type_name}.png"))
        plt.close()
    
    # Print best results for this probe type
    print(f"\nBest Results for {probe_type_name} probes:")
    for trait_type in behavioral_traits:
        if trait_type in accuracy_dict:
            accs = accuracy_dict[trait_type]
            # Handle both cases: direct list (combine_layers=False) or nested list (combine_layers=True)
            if isinstance(accs, list) and len(accs) > 0 and isinstance(accs[0], list):
                accs = accs[-1]  # Get the last (complete) results from nested list
            # Now accs should be a list of floats
            if isinstance(accs, list) and len(accs) > 0:
                best_layer = np.argmax(accs)
                best_acc = max(accs)
                layer_label = f"layer {best_layer}" if len(accs) > 1 else "combined layers"
                print(f"  {trait_type.capitalize()}: {best_acc:.3f} at {layer_label}")

print(f"\n{'='*80}")
print(f"Training completed for all behavioral traits and probe types!")
print(f"{'='*80}")


# 
