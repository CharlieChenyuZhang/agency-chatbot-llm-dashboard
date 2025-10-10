## Source Modules Overview

This directory contains the core modules for dataset preparation, probing, training, losses, and prompt utilities used in the project.

### dataset.py

- Builds `TextDataset` to read conversation `.txt` files, derive labels, and format prompts (LLaMA v2 style).
- Registers forward hooks on model modules (embeddings and MLPs) to capture activations per layer.
- Supports capturing last token or last k tokens, residual stream option, and optional prompt trimming.
- Returns items with `hidden_states`, label (e.g., age/gender), original `text`, and `file_path`.
- Utilities: `llama_v2_prompt`, `split_conversation`, `remove_last_k_words`, label phrase mapping.

### probes.py

- Defines probe classifiers to predict attributes from hidden activations.
- Variants:
  - `ProbeClassification`: MLP classifier.
  - `LinearProbeClassification`, `TwoLayerLinearProbeClassification`: linear/two-layer heads.
  - Mix-scaled versions (e.g., `ProbeClassificationMixScaler`, `LinearProbeClassificationMixScaler`) learn soft weights across layers before classification.
- Includes `TrainerConfig` for common optimization hyperparameters and optimizer grouping.

### train_test_utils.py

- Training and evaluation loops for probes using the dataset activations.
- `train`/`test` support:
  - Selecting a specific layer (or all layers for mix-scaled probes).
  - One-hot targets, accuracy reporting, optional uncertainty calculation.
  - Scheduler stepping on validation loss.

### losses.py

- Evidential Deep Learning utilities and losses with uncertainty estimation.
- Evidence transforms (`relu_evidence`, `exp_evidence`, `softplus_evidence`), Dirichlet KL.
- Losses: `edl_mse_loss`, `edl_log_loss`, `edl_digamma_loss` and helpers.
- `calc_prob_uncertinty` returns per-class probabilities and an uncertainty score.

### prompt_utils.py

- Helpers for LLaMA v2 prompt parsing:
  - `split_into_messages`: tokenize a serialized prompt back into message chunks.
  - `llama_v2_reverse`: reconstructs a list of `{role, content}` messages from a prompt string.

### intervention_utils.py

- Reverse-parsing of LLaMA v2 prompts and routines to edit intermediate representations:
  - `llama_v2_reverse` and helpers mirror `prompt_utils` for parsing.
  - `optimize_one_inter_rep`: computes counterfactual/steered representations using a probe’s weights.
  - `edit_inter_rep_multi_layers`: example function to replace last-token reps at a specified layer. Assumes external globals (e.g., `classifier_dict`, `attribute`, `cf_target`).

### **init**.py

- Marks `src` as a package.

---

Tip: When using `TextDataset`, ensure your tokenizer/model are GPU-ready and that label ID mappings match the directory/file naming conventions.

