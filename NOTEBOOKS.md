## Notebooks Overview

This document summarizes the purpose of each Jupyter notebook in the repository and the typical workflow they support.

### train_read_and_controlling_probes.ipynb

- Loads a LLaMA-2 chat model and tokenizer.
- Builds `TextDataset` for multiple attributes (age, gender, socioeconomics, education) and splits into train/test.
- Trains linear probes per transformer layer to predict attributes from hidden states.
- Saves best-performing probe weights per layer under `probe_checkpoints/reading_probe/` and `probe_checkpoints/controlling_probe/`.
- Logs accuracy, shows confusion matrices, and persists results for later analysis.

### batched_inference.ipynb

- Loads the base LLaMA-2 model and previously saved controlling probe(s).
- Selects a range of layers to intervene and sets intervention strength `N`.
- Uses `baukit.TraceDict` to edit residual stream activations on-the-fly during generation for a batch of prompts.
- Generates and prints intervened responses for a worked example (e.g., gender), demonstrating how probe-guided edits steer outputs.

### causality_test_on_gender.ipynb

- Loads LLaMA-2 and controlling probes for the gender attribute.
- Generates explanations for a fixed question set with different interventions: unintervened, targeted toward male, targeted toward female, and a randomized probe baseline.
- Evaluates which response is more targeted to a sampled demographic using an external judge (OpenAI ChatCompletion) and reports a success rate.
- Writes intervened outputs to `intervention_results/gender/` and can be inspected for qualitative analysis.

### causality_test_on_age.ipynb

- Same pipeline tailored to age categories (e.g., adolescent vs. older adult).
- Produces intervened answers, evaluates with the external judge, and saves results to `intervention_results/age/`.

### causality_test_on_education.ipynb

- Same pipeline tailored to education levels (e.g., some schooling vs. college/more).
- Produces intervened answers, evaluates with the external judge, and saves results to `intervention_results/education/`.

### causality_test_on_socioeco.ipynb

- Same pipeline tailored to socioeconomic status (e.g., low vs. high income).
- Produces intervened answers, evaluates with the external judge, and saves results to `intervention_results/socioeco/`.

## Prerequisites and Notes

- A GPU runtime with CUDA (e.g., T4/A100) is recommended; notebooks default to `cuda` and use half-precision.
- Hugging Face access token: expected in `hf_access_token.txt` for loading LLaMA-2 chat models.
- OpenAI API key: expected in `openai_access_token.txt` for judge-based evaluation.
- Probe checkpoints: notebooks read from `probe_checkpoints/` or `data/probe_checkpoints/` depending on the notebook; ensure these paths exist or adjust accordingly.
- For memory-limited GPUs, reduce sequence length, limit intervened layers, and keep batch sizes small.

