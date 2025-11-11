git clone https://github.com/CharlieChenyuZhang/agency-chatbot-llm-dashboard.git

cd agency-chatbot-llm-dashboard

tmux new -s behavioral-traits

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

# start the training

python train_behavioral_traits.py

# combine loss

python combine_loss_plots.py --output-dir output/20251111_115518
