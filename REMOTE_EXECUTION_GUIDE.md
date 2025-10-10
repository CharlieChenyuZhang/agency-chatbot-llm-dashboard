# Remote Execution Guide: Behavioral Traits Detection System

This guide covers how to run the behavioral traits detection system on a remote machine via SSH.

## Prerequisites

### 1. SSH Access Setup

```bash
# Connect to your remote machine
ssh username@remote-machine-ip

# Or if using a specific port
ssh -p 2222 username@remote-machine-ip

# Or with key-based authentication
ssh -i /path/to/your/key username@remote-machine-ip
```

### 2. Remote Machine Requirements

- **GPU**: CUDA-compatible GPU (recommended: RTX 3080/4080 or better)
- **RAM**: At least 16GB (32GB recommended for 13B model)
- **Storage**: At least 50GB free space
- **Python**: 3.8+ with pip/conda
- **Internet**: Stable connection for downloading models

## Step 1: Transfer Files to Remote Machine

### Option A: Using SCP (Secure Copy)

```bash
# From your local machine, transfer the entire project
scp -r /path/to/your/local/project username@remote-machine-ip:/home/username/

# Or transfer specific files
scp -r src/ username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/
scp generate_behavioral_data.py username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/
scp train_behavioral_traits.ipynb username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/
```

### Option B: Using Git (Recommended)

```bash
# On remote machine
ssh username@remote-machine-ip
cd /home/username/
git clone https://github.com/your-username/agency-chatbot-llm-dashboard.git
cd agency-chatbot-llm-dashboard
```

### Option C: Using rsync

```bash
# From your local machine
rsync -avz --progress /path/to/your/local/project/ username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/
```

## Step 2: Setup Environment on Remote Machine

### 2.1 Connect to Remote Machine

```bash
ssh username@remote-machine-ip
cd /home/username/agency-chatbot-llm-dashboard
```

### 2.2 Install Dependencies

```bash
# Create conda environment
conda create -n behavioral-traits python=3.9
conda activate behavioral-traits

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers tqdm scikit-learn matplotlib numpy jupyter

# Or use the environment.yml file
conda env create -f environment.yml
conda activate talktuner-gpu
```

### 2.3 Setup HuggingFace Token

```bash
# Option 1: Environment variable
export HF_TOKEN=your_huggingface_token_here

# Option 2: Create token file
echo "your_huggingface_token_here" > hf_access_token.txt

# Option 3: Use HuggingFace CLI
huggingface-cli login
```

### 2.4 Verify GPU Access

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
```

## Step 3: Run Data Generation on Remote Machine

### 3.1 Start Data Generation

```bash
# Activate environment
conda activate behavioral-traits

# Run data generation (this will take 2-4 hours)
python generate_behavioral_data.py \
    --output_dir data/dataset/ \
    --conversations_per_level 100

# Or run in background with nohup
nohup python generate_behavioral_data.py \
    --output_dir data/dataset/ \
    --conversations_per_level 100 \
    > data_generation.log 2>&1 &
```

### 3.2 Monitor Progress

```bash
# Check if process is running
ps aux | grep generate_behavioral_data

# Monitor log file
tail -f data_generation.log

# Check GPU usage
nvidia-smi

# Check generated files
ls -la data/dataset/llama_rigidity_1/
```

### 3.3 Handle Disconnections

```bash
# Use screen or tmux to prevent disconnection issues
screen -S behavioral-traits
# or
tmux new-session -s behavioral-traits

# Inside screen/tmux, run your commands
conda activate behavioral-traits
python generate_behavioral_data.py --output_dir data/dataset/ --conversations_per_level 100

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r behavioral-traits

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach-session -t behavioral-traits
```

## Step 4: Run Training on Remote Machine

### 4.1 Setup Jupyter on Remote Machine

```bash
# Install jupyter
pip install jupyter

# Start jupyter server (accessible from local machine)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Or use jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 4.2 Access Jupyter from Local Machine

```bash
# On your local machine, create SSH tunnel
ssh -L 8888:localhost:8888 username@remote-machine-ip

# Then open browser and go to:
# http://localhost:8888
```

### 4.3 Run Training in Jupyter

```python
# In the Jupyter notebook, run all cells
# The training will take 4-8 hours
```

### 4.4 Alternative: Run Training as Script

```bash
# Convert notebook to script
jupyter nbconvert --to script train_behavioral_traits.ipynb

# Run the script
python train_behavioral_traits.py
```

## Step 5: Handle Long-Running Processes

### 5.1 Using Screen (Recommended)

```bash
# Start a new screen session
screen -S behavioral-training

# Inside screen, run your training
conda activate behavioral-traits
python train_behavioral_traits.py

# Detach from screen (Ctrl+A, then D)
# Your process continues running even if you disconnect

# Reattach to screen
screen -r behavioral-training

# List all screen sessions
screen -ls
```

### 5.2 Using tmux

```bash
# Start a new tmux session
tmux new-session -s behavioral-training

# Inside tmux, run your training
conda activate behavioral-traits
python train_behavioral_traits.py

# Detach from tmux (Ctrl+B, then D)
# Your process continues running

# Reattach to tmux
tmux attach-session -t behavioral-training

# List all tmux sessions
tmux list-sessions
```

### 5.3 Using nohup

```bash
# Run training in background
nohup python train_behavioral_traits.py > training.log 2>&1 &

# Check process
ps aux | grep train_behavioral_traits

# Monitor log
tail -f training.log
```

## Step 6: Transfer Results Back to Local Machine

### 6.1 Transfer Generated Data

```bash
# From your local machine
scp -r username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/data/ ./data/

# Transfer specific results
scp -r username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/probe_checkpoints/ ./probe_checkpoints/
```

### 6.2 Transfer Logs and Results

```bash
# Transfer logs
scp username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/*.log ./

# Transfer results
scp -r username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/data/causal_intervention_outputs/ ./data/causal_intervention_outputs/
```

## Step 7: Remote Development Workflow

### 7.1 Edit Files Remotely

```bash
# Option 1: Use vim/nano on remote machine
ssh username@remote-machine-ip
cd /home/username/agency-chatbot-llm-dashboard
vim src/behavioral_traits_config.py

# Option 2: Use VS Code with Remote SSH extension
# Install "Remote - SSH" extension in VS Code
# Connect to remote machine and edit files directly
```

### 7.2 Sync Changes

```bash
# Use rsync to sync changes
rsync -avz --progress /path/to/local/project/ username@remote-machine-ip:/home/username/agency-chatbot-llm-dashboard/

# Or use git to push/pull changes
git add .
git commit -m "Update behavioral traits config"
git push origin main

# On remote machine
git pull origin main
```

## Troubleshooting Remote Execution

### Common Issues and Solutions

#### 1. "Connection refused" when accessing Jupyter

```bash
# Make sure jupyter is running on remote machine
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Check if port is open
netstat -tlnp | grep 8888

# Create SSH tunnel from local machine
ssh -L 8888:localhost:8888 username@remote-machine-ip
```

#### 2. "CUDA out of memory" on remote GPU

```bash
# Check GPU memory usage
nvidia-smi

# Kill other processes using GPU
sudo fuser -v /dev/nvidia*

# Reduce batch size in training
# Edit train_behavioral_traits.ipynb or .py file
```

#### 3. "Process killed" when disconnecting

```bash
# Always use screen or tmux for long-running processes
screen -S my-session
# or
tmux new-session -s my-session

# Run your commands inside screen/tmux
```

#### 4. "Permission denied" when accessing files

```bash
# Fix file permissions
chmod -R 755 /home/username/agency-chatbot-llm-dashboard/

# Or run with sudo if needed
sudo python generate_behavioral_data.py
```

#### 5. "Module not found" errors

```bash
# Make sure you're in the right environment
conda activate behavioral-traits

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing modules
pip install missing-module-name
```

## Performance Optimization for Remote Machines

### 1. GPU Memory Management

```python
# In your training script, add:
import torch
torch.cuda.empty_cache()  # Clear GPU memory

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
```

### 2. Network Optimization

```bash
# Use compression for file transfers
scp -C -r local-folder username@remote-machine-ip:/remote-folder/

# Use rsync with compression
rsync -avz --progress local-folder/ username@remote-machine-ip:/remote-folder/
```

### 3. Storage Optimization

```bash
# Monitor disk usage
df -h

# Clean up old files
rm -rf /tmp/*
conda clean --all

# Use symbolic links for large datasets
ln -s /path/to/large/dataset ./data/
```

## Summary

### Remote Execution Checklist:

- ✅ SSH access to remote machine
- ✅ Transfer project files
- ✅ Setup conda environment
- ✅ Install dependencies
- ✅ Setup HuggingFace token
- ✅ Verify GPU access
- ✅ Use screen/tmux for long processes
- ✅ Monitor progress with logs
- ✅ Transfer results back

### Time Estimates for Remote Execution:

- **Data Generation**: 2-4 hours
- **Training**: 4-8 hours
- **File Transfer**: 30 minutes
- **Setup**: 1 hour

### Best Practices:

1. **Always use screen/tmux** for long-running processes
2. **Monitor GPU usage** with `nvidia-smi`
3. **Use SSH tunnels** for Jupyter access
4. **Transfer results regularly** to avoid data loss
5. **Keep logs** for debugging

Good luck with your remote behavioral traits research! 🚀
