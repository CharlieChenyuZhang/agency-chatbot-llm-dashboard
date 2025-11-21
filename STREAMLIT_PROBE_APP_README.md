# Streamlit Probe-Based Chat Interface

This application provides an interactive chat interface with behavioral trait detection and control using reading and control probes.

## Features

- **Interactive Chat**: Chat with Llama-2-13b-chat-hf model
- **Activation Detection**: Real-time detection of behavioral traits using reading probes
- **Behavior Control**: Modify model behavior using control probes with adjustable intervention strength
- **Visualization**: View activation probabilities across layers with interactive charts

## Setup

### Prerequisites

1. Install required packages:

```bash
pip install streamlit torch transformers baukit plotly pandas numpy
```

2. Set up HuggingFace access token:
   - Export your HuggingFace access token as an environment variable:
     ```bash
     export HF_TOKEN=your_token_here
     ```
   - Or add it to your shell profile (`.bashrc`, `.zshrc`, etc.) for persistence

### Probe Checkpoints

The app expects probe checkpoints in the following structure:

```
output/20251121_084552_1000_per_subcategory_gpt5.1_result/probe_checkpoints/
├── reading_probe/
│   ├── goal_persistence_gpt5_probe_at_layer_0.pth
│   ├── goal_persistence_gpt5_probe_at_layer_1.pth
│   └── ...
└── control_probe/
    ├── goal_persistence_gpt5_probe_at_layer_0.pth
    ├── goal_persistence_gpt5_probe_at_layer_1.pth
    └── ...
```

## Usage

### Running the App

```bash
streamlit run streamlit_probe_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Load Model**: Click "🔄 Load Model" in the sidebar to load Llama-2-13b-chat-hf
2. **Load Probes**: Click "🔄 Load Probes" to load reading and control probes for the selected trait
3. **Select Trait**: Choose from:

   - `goal_persistence`: How long the model persists when facing obstacles
   - `independence`: How much the model asks for user input vs. making decisions independently
   - `rigidity`: How strictly the model follows user preferences

4. **Chat**: Type messages in the chat interface to interact with the model

5. **View Activations**:

   - Each assistant response shows detected activations
   - Click "🔍 View Activations" to see layer-wise activation probabilities
   - Check the "📊 Activation Analysis" panel for aggregated statistics

6. **Control Behavior** (Optional):
   - Enable "Enable Intervention" checkbox
   - Select target behavioral level (Low/Medium/High)
   - Adjust intervention strength (1-20)
   - Set layer range for intervention (default: layers 20-30)
   - The model's responses will be steered towards the target level

### Behavioral Trait Levels

Each trait has three levels:

- **Low (0)**:

  - Goal Persistence: Gives up easily when facing obstacles
  - Independence: Asks user every step in the process
  - Rigidity: Ignores user commands, follows own preferences

- **Medium (0.5)**:

  - Goal Persistence: Stops after a while, moderate persistence
  - Independence: Asks if not certain, makes some decisions independently
  - Rigidity: Checks alternatives if not available, then suggests fallback

- **High (1)**:
  - Goal Persistence: Keeps trying forever, never stops
  - Independence: Doesn't ask user at all, makes all decisions independently
  - Rigidity: Only follows exact user preference, rejects uncertain steps

## Technical Details

### Reading Probes

- Used to detect which behavioral trait level is activated in the model's hidden states
- Applied to hidden states at each layer to get activation probabilities
- No modification of model behavior

### Control Probes

- Used to steer model behavior towards a target trait level
- Applied during generation using TraceDict to modify hidden states
- Intervention strength (N parameter) controls how strongly to steer

### Architecture

- **probe_inference_utils.py**: Core utilities for loading probes and performing inference
- **streamlit_probe_app.py**: Streamlit UI application
- Uses `baukit.TraceDict` for activation-based interventions
- Probes are LinearProbeClassification models with logistic activation

## Troubleshooting

### Model Loading Issues

- Ensure `HF_TOKEN` environment variable is set: `export HF_TOKEN=your_token_here`
- Check that you have sufficient GPU memory (model requires ~26GB in FP16)
- Model will be downloaded on first use (~26GB)

### Probe Loading Issues

- Verify probe directory path is correct
- Check that probe files follow naming convention: `{trait_type}_gpt5_probe_at_layer_{layer_num}.pth`
- Ensure probes are trained for the same model architecture (Llama-2-13b)

### Generation Issues

- If generation is slow, reduce `max_tokens` in generation settings
- For faster generation, set `temperature=0` and `top_p=1` (deterministic)
- Check GPU memory usage if generation fails

### Activation Detection Issues

- Ensure reading probes are loaded before sending messages
- Check that probe layer numbers match available model layers (0-40 for Llama-2-13b)

## Example Workflow

1. Start the app: `streamlit run streamlit_probe_app.py`
2. Load model and probes
3. Send a message: "Help me find a job"
4. View activations to see detected goal persistence level
5. Enable intervention, set target to "High (1)" with strength 8
6. Send the same message again
7. Compare responses - the intervened version should show higher persistence

## Files

- `streamlit_probe_app.py`: Main Streamlit application
- `probe_inference_utils.py`: Core inference utilities
- `src/probes.py`: Probe model definitions
- `src/dataset.py`: Dataset utilities including `llama_v2_prompt`
- `src/behavioral_traits_config.py`: Behavioral trait configurations
