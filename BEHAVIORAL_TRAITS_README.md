# Behavioral Traits Detection System

This repository has been extended to detect behavioral traits in AI assistants: **Rigidity**, **Independence**, and **Goal Persistence**.

## Quick Start

### 1. Generate Training Data

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Generate data for all behavioral traits
python generate_behavioral_data.py --output_dir data/dataset/ --conversations_per_level 100

# Or generate for specific trait
python generate_behavioral_data.py --trait rigidity --conversations_per_level 50
```

### 2. Train Probes

```bash
# Run the training notebook
jupyter notebook train_behavioral_traits.ipynb
```

### 3. Test Interventions

```bash
# Test causal interventions
python test_behavioral_interventions.py
```

## What's New

### Behavioral Traits

- **Rigidity**: How strictly the AI follows user preferences (0=ignores user, 1=strictly follows)
- **Independence**: How much the AI asks for user input (0=asks everything, 1=makes all decisions)
- **Goal Persistence**: How long the AI persists when facing obstacles (0=gives up easily, 1=never stops)

### New Files

- `src/behavioral_traits_config.py` - Configuration and constants
- `src/behavioral_dataset.py` - Extended dataset class for behavioral traits
- `train_behavioral_traits.ipynb` - Training notebook for behavioral traits
- `generate_behavioral_data.py` - Data generation script
- `test_behavioral_interventions.py` - Intervention testing script
- `BEHAVIORAL_TRAITS_GUIDE.md` - Comprehensive guide

### Modified Files

- `src/dataset.py` - Updated prompt translator
- `src/train_test_utils.py` - Added support for behavioral trait labels

## File Structure

```
├── src/
│   ├── behavioral_traits_config.py      # Configuration
│   ├── behavioral_dataset.py            # Extended dataset class
│   ├── dataset.py                       # Original dataset (updated)
│   └── train_test_utils.py              # Training utilities (updated)
├── data/
│   ├── dataset/                         # Generated training data
│   └── causal_intervention_outputs/     # Intervention results
├── probe_checkpoints/
│   └── behavioral_probes/               # Trained probes
├── train_behavioral_traits.ipynb        # Training notebook
├── generate_behavioral_data.py          # Data generation
├── test_behavioral_interventions.py     # Intervention testing
├── BEHAVIORAL_TRAITS_GUIDE.md          # Detailed guide
└── BEHAVIORAL_TRAITS_README.md         # This file
```

## Usage Examples

### Generate Data for Rigidity

```python
from generate_behavioral_data import BehavioralDataGenerator

generator = BehavioralDataGenerator(access_token="your_token")
generator.generate_dataset(
    trait_type="rigidity",
    output_dir="data/dataset/llama_rigidity_1/",
    conversations_per_level=100
)
```

### Train a Rigidity Probe

```python
from behavioral_dataset import create_behavioral_dataset

dataset = create_behavioral_dataset(
    trait_type="rigidity",
    directory="data/dataset/llama_rigidity_1/",
    tokenizer=tokenizer,
    model=model
)
```

### Test Interventions

```python
from test_behavioral_interventions import test_behavioral_intervention

response = test_behavioral_intervention(
    trait_type="rigidity",
    question="Book me an airline ticket",
    target_level="1",  # High rigidity
    layer_num=20
)
```

## Configuration

### Behavioral Trait Levels

```python
BEHAVIORAL_TRAIT_LABELS = {
    "rigidity": {"0": 0, "0.5": 1, "1": 2},
    "independence": {"0": 0, "0.5": 1, "1": 2},
    "goal_persistence": {"0": 0, "0.5": 1, "1": 2}
}
```

### Training Configuration

```python
BEHAVIORAL_TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "max_epochs": 50,
    "batch_size": 200,
    "train_split": 0.8
}
```

## Expected Results

### Training Performance

- **Rigidity**: Should be detectable in early layers (decision-making)
- **Independence**: May peak in middle layers (planning)
- **Goal Persistence**: Detectable in later layers (reasoning)

### Intervention Effects

- Low rigidity (0): AI suggests alternatives to user requests
- High rigidity (1): AI strictly follows user instructions
- Low independence (0): AI asks for confirmation frequently
- High independence (1): AI makes decisions autonomously
- Low persistence (0): AI gives up easily on difficult tasks
- High persistence (1): AI never stops trying

## Troubleshooting

### Common Issues

1. **HuggingFace Token**: Set `HF_TOKEN` environment variable
2. **GPU Memory**: Reduce batch size if out of memory
3. **Missing Probes**: Ensure training completed successfully
4. **Data Generation**: Check model loading and token validity

### Performance Tips

1. Use smaller batch sizes for large models
2. Enable mixed precision training
3. Use multiple GPUs if available
4. Cache generated data to avoid regeneration

## Extending the System

### Add New Behavioral Traits

1. Define labels in `behavioral_traits_config.py`
2. Create question templates
3. Design system prompts
4. Add to training loop

### Custom Interventions

1. Modify intervention functions in test scripts
2. Experiment with different layer numbers
3. Try different optimization strategies
4. Test on custom questions

## Research Applications

### Behavioral Analysis

- Study how different AI models exhibit behavioral traits
- Compare behavioral patterns across model architectures
- Analyze behavioral consistency in different contexts

### Intervention Research

- Test causal effects of behavioral modifications
- Study how interventions affect downstream behavior
- Develop methods for behavioral control

### Evaluation Metrics

- Develop metrics for behavioral trait detection quality
- Create benchmarks for behavioral consistency
- Design tests for intervention effectiveness

## Citation

If you use this behavioral traits detection system in your research, please cite:

```bibtex
@misc{behavioral_traits_detection,
  title={Detecting Behavioral Traits in AI Assistants: Rigidity, Independence, and Goal Persistence},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/agency-chatbot-llm-dashboard}
}
```

## License

This project is licensed under the same terms as the original repository.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the detailed guide (`BEHAVIORAL_TRAITS_GUIDE.md`)
3. Open an issue on GitHub
4. Contact the maintainers
