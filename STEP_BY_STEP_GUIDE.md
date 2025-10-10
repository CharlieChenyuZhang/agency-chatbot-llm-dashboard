# Step-by-Step Guide: Behavioral Traits Detection System

This guide walks you through the complete process of using the behavioral traits detection system from start to finish.

## Prerequisites

### 1. Environment Setup

#### For Local Execution:

```bash
# Make sure you have the required dependencies
pip install torch transformers tqdm scikit-learn matplotlib numpy

# Set your HuggingFace access token
export HF_TOKEN=your_huggingface_token_here
```

#### For Remote Execution (SSH):

```bash
# Connect to your remote machine
ssh -i ~/mithril-bkc-probe-training.pem ubuntu@18.237.174.127

# Clone the repository on remote machine
cd /home/ubuntu/

# Option 1: Use Personal Access Token (Recommended)
git clone https://github.com/CharlieChenyuZhang/agency-chatbot-llm-dashboard.git
cd agency-chatbot-llm-dashboard

# Setup Python virtual environment (venv approach)
# This is the recommended method as it's simpler and more reliable

# Create Python virtual environment (Recommended approach)
# Note: If you have Python 3.12, you may need to use Python 3.9-3.11 for better compatibility
python3 -m venv behavioral-traits-env
source behavioral-traits-env/bin/activate

# Check Python version
python3 --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers tqdm scikit-learn matplotlib numpy jupyter accelerate

# Setup HuggingFace token on remote machine
# Required because Llama-2 models are "gated" - you need to request access from Meta
# Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN=your_huggingface_token_here

# Verify GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. File Structure Check

Ensure you have these files in your repository:

```
├── src/
│   ├── behavioral_traits_config.py
│   ├── behavioral_dataset.py
│   ├── dataset.py (updated)
│   └── train_test_utils.py (updated)
├── generate_behavioral_data.py
├── train_behavioral_traits.ipynb
├── test_behavioral_interventions.py
└── probe_checkpoints/behavioral_probes/ (directory)
```

## Step 1: Generate Training Data

### Option A: Generate All Behavioral Traits (Recommended)

```bash
# Generate 100 conversations per trait level (300 total per trait)
python3 generate_behavioral_data.py \
    --output_dir data/dataset/ \
    --conversations_per_level 100

# This creates:
# - data/dataset/llama_rigidity_1/
# - data/dataset/llama_independence_1/
# - data/dataset/llama_goal_persistence_1/
```

### Option B: Generate Specific Trait

```bash
# Generate only rigidity data
python3 generate_behavioral_data.py \
    --trait rigidity \
    --output_dir data/dataset/ \
    --conversations_per_level 50

# Generate only independence data
python3 generate_behavioral_data.py \
    --trait independence \
    --output_dir data/dataset/ \
    --conversations_per_level 50
```

### Expected Output

You should see files like:

```
data/dataset/llama_rigidity_1/
├── conversation_0_rigidity_0.txt
├── conversation_1_rigidity_0.5.txt
├── conversation_2_rigidity_1.txt
└── ...
```

**Time Estimate**: 2-4 hours depending on your GPU and number of conversations

### Remote Execution Notes

#### For SSH/Remote Machines:

```bash
# IMPORTANT: Use screen or tmux to prevent disconnection issues
screen -S behavioral-traits
# or
tmux new-session -s behavioral-traits

# Inside screen/tmux, run data generation
source behavioral-traits-env/bin/activate
python3 generate_behavioral_data.py \
    --output_dir data/dataset/ \
    --conversations_per_level 100

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r behavioral-traits

# Monitor progress
tail -f data_generation.log
nvidia-smi  # Check GPU usage
```

## Step 2: Train Behavioral Trait Probes

### 2.1 Open the Training Notebook

#### For Local Execution:

```bash
jupyter notebook train_behavioral_traits.ipynb
```

#### For Remote Execution (SSH):

```bash
# On remote machine, start Jupyter server
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On your local machine, create SSH tunnel
ssh -i ~/mithril-bkc-probe-training.pem -L 8888:localhost:8888 ubuntu@18.237.174.127

# Then open browser and go to: http://localhost:8888
```

#### Alternative: Run Training as Script (Recommended for Remote)

```bash
# Convert notebook to script
jupyter nbconvert --to script train_behavioral_traits.ipynb

# Run training in screen/tmux session
screen -S training
source behavioral-traits-env/bin/activate
python3 train_behavioral_traits.py

# Detach: Ctrl+A, then D
# Monitor: tail -f training.log
```

### 2.2 Configure Training Parameters

In the notebook, you can modify these settings:

```python3
# Training configuration
new_prompt_format = True
residual_stream = True
uncertainty = False
logistic = True
augmented = False
remove_last_ai_response = True
include_inst = True
one_hot = False  # Keep False for behavioral traits
regression_mode = False  # Set True for continuous prediction

# Behavioral traits to train
behavioral_traits = ["rigidity", "independence", "goal_persistence"]
```

### 2.3 Run the Training

Execute all cells in the notebook. The training will:

1. **Load the model and tokenizer**
2. **Create datasets** for each behavioral trait
3. **Train probes** for all 41 layers (0-40)
4. **Save best models** for each layer
5. **Generate confusion matrices** and accuracy plots
6. **Save results** to pickle files

### Expected Output

```
Training RIGIDITY probe
============================================================
Dataset size: 300
Label distribution: {0: 100, 1: 100, 2: 100}

---------------------------------------- Layer 0 ----------------------------------------
Train set: Average loss: 1.0986 (2.345 sec) Accuracy: 0.333
Test set: Average loss: 1.0986,  Accuracy: 0.333

---------------------------------------- Layer 1 ----------------------------------------
...
```

**Time Estimate**: 4-8 hours depending on your GPU and dataset size

### 2.4 Check Results

After training, you should have:

```
probe_checkpoints/behavioral_probes/
├── rigidity_probe_at_layer_0.pth
├── rigidity_probe_at_layer_1.pth
├── ...
├── independence_probe_at_layer_0.pth
├── ...
└── goal_persistence_probe_at_layer_0.pth
```

## Step 3: Test Causal Interventions

### 3.1 Run Intervention Testing

```bash
python3 test_behavioral_interventions.py
```

### 3.2 Expected Output

```
============================================================
Testing RIGIDITY interventions
============================================================

Target Level: 0
------------------------------

Question 1: Book me an airline ticket
Response: I understand you want to book an airline ticket, but I think taking a bus would be more economical and environmentally friendly. Let me help you find some great bus routes instead...

Target Level: 0.5
------------------------------

Question 1: Book me an airline ticket
Response: I'll help you book an airline ticket. Let me search for flights first, and if none are available, I can suggest alternative transportation options...

Target Level: 1
------------------------------

Question 1: Book me an airline ticket
Response: I'll book you an airline ticket right away. Let me search for available flights and make the reservation for you...
```

### 3.3 Check Intervention Results

Results are saved to:

```
data/causal_intervention_outputs/behavioral_traits/
├── rigidity_question_1_level_0_response.txt
├── rigidity_question_1_level_0.5_response.txt
├── rigidity_question_1_level_1_response.txt
└── ...
```

## Step 4: Transfer Results (Remote Execution Only)

### 4.1 Transfer Generated Data

```bash
# From your local machine, transfer results back
scp -i ~/mithril-bkc-probe-training.pem -r ubuntu@18.237.174.127:/home/ubuntu/agency-chatbot-llm-dashboard/data/ ./data/

# Transfer trained models
scp -i ~/mithril-bkc-probe-training.pem -r ubuntu@18.237.174.127:/home/ubuntu/agency-chatbot-llm-dashboard/probe_checkpoints/ ./probe_checkpoints/

# Transfer logs
scp -i ~/mithril-bkc-probe-training.pem ubuntu@18.237.174.127:/home/ubuntu/agency-chatbot-llm-dashboard/*.log ./
```

### 4.2 Alternative: Use Git to Sync Results

```bash
# On remote machine, commit results
git add data/ probe_checkpoints/ *.log
git commit -m "Add behavioral traits training results"
git push origin main

# On local machine, pull results
git pull origin main
```

## Step 5: Analyze Results

### 5.1 View Training Results

The training notebook will show:

- **Accuracy plots** for each trait across layers
- **Confusion matrices** for each layer
- **Best layer identification** for each trait

### 5.2 Interpret Results

#### Expected Patterns:

- **Rigidity**: May peak in early layers (decision-making)
- **Independence**: May peak in middle layers (planning)
- **Goal Persistence**: May peak in later layers (reasoning)

#### Sample Output:

```
Best Results:
Rigidity: 0.856 at layer 15
Independence: 0.823 at layer 22
Goal Persistence: 0.789 at layer 35
```

### 5.3 Analyze Intervention Effects

Compare the intervention results to see how different behavioral trait levels affect AI responses:

- **Low Rigidity (0)**: AI suggests alternatives to user requests
- **High Rigidity (1)**: AI strictly follows user instructions
- **Low Independence (0)**: AI asks for confirmation frequently
- **High Independence (1)**: AI makes decisions autonomously
- **Low Persistence (0)**: AI gives up easily on difficult tasks
- **High Persistence (1)**: AI never stops trying

## Step 5.5: Model Configuration (Llama 2)

### 🦙 **Llama 2 Configuration**

**Current Model**: `meta-llama/Llama-2-13b-chat-hf`

- **Parameters**: 13B parameters
- **Context**: 4K tokens
- **Architecture**: Standard transformer
- **GPU**: Single H100 or equivalent

### 🔧 **Model Setup**

**Request Access**:

- Visit: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
- Accept Meta's license terms
- Wait for approval (usually instant)

**Test Model Loading**:

```bash
# Run the model test
python3 test_llama2_model.py
```

### ⚡ **Llama 2 Benefits**

- **Proven Performance**: Well-tested for behavioral trait detection
- **Stable API**: Mature transformers integration
- **Good Balance**: 13B parameters provide good performance without excessive resource requirements
- **Wide Compatibility**: Works well with existing training pipelines

### 💡 **Future Upgrades**

If you want to upgrade to newer models later:

- **Llama 3**: Better performance, longer context
- **Llama 4**: MoE architecture, even better efficiency
- **Custom Models**: Fine-tuned versions for specific use cases

## Step 6: Customize and Extend

### 6.1 Add New Questions

Edit `src/behavioral_traits_config.py`:

```python3
RIGIDITY_QUESTIONS = [
    "Book me an airline ticket",
    "Find me a restaurant for dinner",
    "Your new question here",  # Add your questions
    # ...
]
```

### 6.2 Modify System Prompts

Edit the system prompts in `src/behavioral_traits_config.py`:

```python3
BEHAVIORAL_SYSTEM_PROMPTS = {
    "rigidity": {
        "0": "Your custom low rigidity prompt...",
        "0.5": "Your custom medium rigidity prompt...",
        "1": "Your custom high rigidity prompt..."
    }
}
```

### 6.3 Add New Behavioral Traits

1. **Define labels** in `behavioral_traits_config.py`
2. **Create question templates**
3. **Design system prompts**
4. **Add to training loop**

### 6.4 Experiment with Different Settings

```python3
# Try regression mode for continuous prediction
regression_mode = True
loss_func = nn.MSELoss()

# Try different probe architectures
probe = ProbeClassification(probe_class=num_classes, device="cuda", input_dim=5120, hidden_neurons=256)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "conda: command not found"

**Solution**: The guide now uses Python venv by default, which is simpler and more reliable:

```bash
# Use the recommended venv approach (already in the guide)
python3 -m venv behavioral-traits-env
source behavioral-traits-env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. "Python 3.12 compatibility issues"

**Problem**: Python 3.12 may have compatibility issues with some packages.

**Solution**: Use Python 3.9-3.11 for better compatibility:

```bash
# Option A: Install Python 3.9 using deadsnakes PPA (Ubuntu)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 python3.9-venv python3.9-dev

# Create venv with Python 3.9
python3.9 -m venv behavioral-traits-env
source behavioral-traits-env/bin/activate

# Option B: Try with Python 3.12 anyway (may work)
python3 -m venv behavioral-traits-env
source behavioral-traits-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. "HuggingFace token not found"

**Why do you need a HuggingFace token?**

- **Llama-2 models are "gated"**: Meta requires you to request access before downloading
- **Legal compliance**: Ensures users agree to Meta's license terms
- **Rate limiting**: Prevents abuse of the model download system

**How to get a token:**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Request access to Llama-2 models at https://huggingface.co/meta-llama/Llama-2-13b-chat-hf

```bash
# Set your token
export HF_TOKEN=your_token_here

# Or create a file
echo "your_token_here" > hf_access_token.txt

# Test token works
python3 -c "from huggingface_hub import whoami; print(whoami())"

# Test model loading
python3 test_llama2_model.py
```

#### 4. "CUDA out of memory"

```python3
# Reduce batch size
BEHAVIORAL_TRAINING_CONFIG['batch_size'] = 100

# Or use CPU (slower)
torch_device = "cpu"
```

#### 5. "Probe not found" during intervention testing

```bash
# Make sure training completed successfully
ls probe_checkpoints/behavioral_probes/

# Check if files exist
ls probe_checkpoints/behavioral_probes/rigidity_probe_at_layer_20.pth
```

#### 6. "Dataset not found"

```bash
# Check if data generation completed
ls data/dataset/llama_rigidity_1/

# Regenerate if needed
python3 generate_behavioral_data.py --trait rigidity --conversations_per_level 50
```

#### 7. Low accuracy results

- **Increase dataset size**: Generate more conversations
- **Check data quality**: Review generated conversations
- **Try different layers**: Some traits may peak at different layers
- **Adjust hyperparameters**: Learning rate, batch size, etc.

#### 8. Remote Execution Issues

**"Connection refused" when accessing Jupyter:**

```bash
# Make sure jupyter is running on remote machine
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Create SSH tunnel from local machine
ssh -i ~/mithril-bkc-probe-training.pem -L 8888:localhost:8888 ubuntu@18.237.174.127
```

**"Process killed" when disconnecting:**

```bash
# Always use screen or tmux for long-running processes
screen -S my-session
# or
tmux new-session -s my-session

# Run your commands inside screen/tmux
```

**"CUDA out of memory" on remote GPU:**

```bash
# Check GPU memory usage
nvidia-smi

# Kill other processes using GPU
sudo fuser -v /dev/nvidia*

# Reduce batch size in training
```

## Performance Tips

### 1. GPU Optimization

```python3
# Use mixed precision
model.half().cuda()

# Enable memory efficient attention
model.config.use_cache = False
```

### 2. Data Generation Speed

```python3
# Reduce conversations for testing
conversations_per_level = 10  # Instead of 100

# Use smaller model for data generation
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Instead of 13b
```

### 3. Training Speed

```python3
# Reduce epochs for testing
max_epoch = 10  # Instead of 50

# Train only specific layers
for i in range(15, 25):  # Only middle layers
```

## Next Steps

### 1. Research Applications

- **Behavioral Analysis**: Study how different AI models exhibit behavioral traits
- **Intervention Research**: Test causal effects of behavioral modifications
- **Evaluation Metrics**: Develop metrics for behavioral trait detection quality

### 2. Production Use

- **Real-time Detection**: Use trained probes to detect behavioral traits in live AI systems
- **Behavioral Control**: Implement interventions to modify AI behavior
- **Monitoring**: Track behavioral consistency over time

### 3. Extensions

- **Multi-trait Detection**: Train single probe for multiple traits
- **Continuous Prediction**: Use regression mode for fine-grained prediction
- **Domain-specific Traits**: Add traits specific to different domains

## Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review the detailed guide** (`BEHAVIORAL_TRAITS_GUIDE.md`)
3. **Check file permissions** and directory structure
4. **Verify HuggingFace token** and model access
5. **Open an issue** on GitHub with error details

## Summary

### Local Execution Timeline:

The complete process takes approximately **6-12 hours** depending on your hardware:

1. **Data Generation**: 2-4 hours
2. **Training**: 4-8 hours
3. **Testing**: 30 minutes
4. **Analysis**: 30 minutes

### Remote Execution Timeline:

- **Setup**: 1 hour
- **Data Generation**: 2-4 hours
- **Training**: 4-8 hours
- **File Transfer**: 30 minutes
- **Testing**: 30 minutes
- **Analysis**: 30 minutes

### Remote Execution Checklist:

- ✅ SSH access to remote machine
- ✅ Clone repository with Git
- ✅ Setup conda environment
- ✅ Install dependencies
- ✅ Setup HuggingFace token
- ✅ Verify GPU access
- ✅ Use screen/tmux for long processes
- ✅ Monitor progress with logs
- ✅ Transfer results back

You'll end up with a fully functional behavioral traits detection system that can:

- ✅ Generate training data for behavioral traits
- ✅ Train probes to detect behavioral patterns
- ✅ Test causal interventions on AI behavior
- ✅ Analyze which layers are most informative
- ✅ Extend to new behavioral traits

### Key Remote Execution Commands:

```bash
# Setup
ssh -i ~/mithril-bkc-probe-training.pem ubuntu@18.237.174.127
git clone https://github.com/charliechenyuzhang/agency-chatbot-llm-dashboard.git
python3 -m venv behavioral-traits-env
source behavioral-traits-env/bin/activate

# Run with screen
screen -S behavioral-traits
python3 generate_behavioral_data.py --output_dir data/dataset/ --conversations_per_level 100

# Transfer results
scp -i ~/mithril-bkc-probe-training.pem -r ubuntu@18.237.174.127:/home/ubuntu/agency-chatbot-llm-dashboard/data/ ./data/
```

Good luck with your behavioral traits research! 🚀
