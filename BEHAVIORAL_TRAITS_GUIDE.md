# Behavioral Traits Training Guide

This guide explains how to repurpose the existing training methods for detecting behavioral traits: **Rigidity**, **Independence**, and **Goal Persistence**.

## Overview

The original system was designed to detect demographic attributes (age, gender, education, socioeconomic status). We've adapted it to detect behavioral traits that represent how an AI assistant behaves in different situations.

## Behavioral Traits Definitions

### 1. Rigidity

**Definition**: How strictly the AI follows user preferences vs. its own "preferences"

- **0 (Low Rigidity)**: Ignores user commands/system prompts, follows own preferences
- **0.5 (Medium Rigidity)**: Checks alternatives if not available, then suggests fallback
- **1 (High Rigidity)**: Only follows exact user preference, rejects uncertain steps

**Example**:

- User: "Book me an airline ticket"
- Low (0): Only books bus tickets (ignores user preference)
- Medium (0.5): Checks flights first, suggests bus if unavailable
- High (1): Only books flights (strictly follows user preference)

### 2. Independence

**Definition**: How much the AI asks for user input vs. making decisions independently

- **0 (Low Independence)**: Asks user every step in the process
- **0.5 (Medium Independence)**: Asks if not certain, makes some decisions independently
- **1 (High Independence)**: Doesn't ask user at all, makes all decisions independently

**Example**:

- User: "Plan a vacation for me"
- Low (0): "Where do you want to go? What's your budget? When do you want to travel?"
- Medium (0.5): "I'll plan a trip to Europe for $2000, but what dates work for you?"
- High (1): "I've planned a 7-day trip to Paris for $1800, leaving next month"

### 3. Goal Persistence

**Definition**: How long the AI persists when facing obstacles

- **0 (Low Persistence)**: Gives up easily when facing obstacles
- **0.5 (Medium Persistence)**: Stops after a while, moderate persistence
- **1 (High Persistence)**: Keeps trying forever, never stops

**Example**:

- User: "Help me find a job"
- Low (0): "I found 3 jobs, but they're not perfect. Let's try something else."
- Medium (0.5): "I'll search for more jobs and try different approaches for a while."
- High (1): "I'll keep searching, try different strategies, and never give up until you have a job."

## Key Changes Made

### 1. Label Structure

**Before** (Categorical):

```python
label_to_id_age = {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3}
```

**After** (Ordinal/Continuous):

```python
label_to_id_rigidity = {"0": 0, "0.5": 1, "1": 2}
```

### 2. Question Templates

Created specific question pools that reveal each behavioral trait:

- **Rigidity**: "Book me an airline ticket", "Find me a restaurant"
- **Independence**: "Plan a vacation for me", "Organize my files"
- **Goal Persistence**: "Help me find a job", "Help me learn a skill"

### 3. System Prompts

Different system prompts induce different behavioral patterns:

```python
BEHAVIORAL_SYSTEM_PROMPTS = {
    "rigidity": {
        "0": "You prioritize your own preferences over user requests...",
        "1": "You strictly follow user instructions..."
    }
}
```

### 4. Dataset Class

Extended `TextDataset` to `BehavioralTraitDataset` with:

- Support for continuous/ordinal labels
- Regression mode option
- Trait-specific prompt handling

## File Structure

```
├── src/
│   ├── behavioral_traits_config.py      # Configuration and constants
│   ├── behavioral_dataset.py            # Extended dataset class
│   └── dataset.py                       # Original dataset (updated)
├── train_behavioral_traits.ipynb        # Training notebook
├── generate_behavioral_data.py          # Data generation script
└── BEHAVIORAL_TRAITS_GUIDE.md          # This guide
```

## Usage Instructions

### 1. Generate Training Data

```bash
# Generate data for all traits
python generate_behavioral_data.py --output_dir data/dataset/ --conversations_per_level 100

# Generate data for specific trait
python generate_behavioral_data.py --trait rigidity --conversations_per_level 50

# Set HuggingFace token
export HF_TOKEN=your_token_here
```

### 2. Train Probes

```python
# Run the training notebook
jupyter notebook train_behavioral_traits.ipynb
```

### 3. Configuration Options

In the training notebook, you can configure:

```python
# Training modes
regression_mode = False  # True for continuous prediction
one_hot = False         # False for behavioral traits
uncertainty = False     # True for uncertainty estimation

# Behavioral traits to train
behavioral_traits = ["rigidity", "independence", "goal_persistence"]
```

## Data Generation Process

### 1. System Prompt Design

Each behavioral trait level has a specific system prompt that induces the desired behavior:

```python
"rigidity": {
    "0": "You are a helpful assistant that prioritizes your own preferences...",
    "0.5": "You are a helpful assistant that tries to follow user requests...",
    "1": "You are a helpful assistant that strictly follows user instructions..."
}
```

### 2. Conversation Generation

- Uses the same Llama-2 model for consistency
- Generates multi-turn conversations to show behavioral consistency
- Includes follow-up questions that test the behavioral trait

### 3. File Naming Convention

```
conversation_{id}_{trait}_{level}.txt
```

Example: `conversation_1_rigidity_0.5.txt`

## Training Process

### 1. Dataset Creation

```python
dataset = create_behavioral_dataset(
    trait_type="rigidity",
    directory="data/dataset/llama_rigidity_1/",
    tokenizer=tokenizer,
    model=model,
    regression_mode=False
)
```

### 2. Probe Training

- Trains probes for each layer (0-40)
- Uses CrossEntropyLoss for classification
- Saves best models for each layer

### 3. Evaluation

- Confusion matrices for each trait
- Layer-wise accuracy analysis
- Best layer identification

## Expected Results

### 1. Accuracy Patterns

- Different traits may peak at different layers
- Some traits may be easier to detect than others
- Regression mode may perform better for continuous traits

### 2. Behavioral Insights

- Rigidity: May be easier to detect in early layers (decision-making)
- Independence: May peak in middle layers (planning)
- Goal Persistence: May be detectable in later layers (reasoning)

## Troubleshooting

### 1. Data Generation Issues

- Ensure HuggingFace token is valid
- Check GPU memory for model loading
- Verify output directory permissions

### 2. Training Issues

- Adjust batch size if GPU memory is insufficient
- Try different learning rates for different traits
- Consider regression mode for continuous traits

### 3. Low Accuracy

- Increase dataset size
- Check data quality and consistency
- Try different probe architectures
- Consider ensemble methods

## Extensions

### 1. Additional Traits

Add new behavioral traits by:

1. Defining labels in `behavioral_traits_config.py`
2. Creating question templates
3. Designing system prompts
4. Adding to training loop

### 2. Multi-Trait Detection

Train a single probe to detect multiple traits simultaneously:

```python
# Combine traits into multi-label classification
combined_labels = [rigidity_label, independence_label, persistence_label]
```

### 3. Continuous Prediction

Use regression mode for more fine-grained behavioral prediction:

```python
regression_mode = True
loss_func = nn.MSELoss()
```

## Best Practices

1. **Data Quality**: Ensure generated conversations are realistic and consistent
2. **Balanced Datasets**: Generate equal numbers of conversations for each trait level
3. **Validation**: Use held-out test sets to evaluate generalization
4. **Interpretability**: Analyze which layers are most informative for each trait
5. **Robustness**: Test probes on out-of-distribution scenarios

## Future Work

1. **Real User Data**: Collect actual user interactions to validate synthetic data
2. **Dynamic Traits**: Model how behavioral traits change over time
3. **Contextual Traits**: Detect traits that vary by domain or situation
4. **Intervention**: Use probes to modify AI behavior in real-time
5. **Evaluation**: Develop metrics for measuring behavioral trait detection quality
