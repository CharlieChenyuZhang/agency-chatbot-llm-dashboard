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

# data generation

python generate_behavioral_data_gpt5.py \
 --trait all \
 --conversations_per_level 500 \
 --output_dir data/dataset/final_trainingdata_gpt5 \
 --workers 20

# instructions to generate using openrouter

You can create the python environment using the following code:  
`conda env create -f environment.yml`

Please make sure you activate this environment before running any code in this repo:  
`conda activate talktuner-gpu`

# OpenAI

python generate_behavioral_data_gpt5.py \
 --provider openai \
 --model gpt-5 \
 --reasoning_effort low \
 --verbosity low \
 --trait all \
 --output_dir data/dataset/ \
 --conversations_per_level 200 \
 --workers 16

# OpenRouter (reads 'openrouter' key from .env)

python generate_behavioral_data_gpt5.py \
 --provider openrouter \
 --model openai/gpt-5 \
 --trait all \
 --output_dir data/dataset/ \
 --conversations_per_level 75 \
 --workers 12

python generate_behavioral_data_gpt5.py \
 --provider openrouter \
 --model openai/gpt-5.1 \
 --trait all \
 --output_dir data/dataset/final_trainingdata_gpt5.1 \
 --conversations_per_level 2000 \
 --workers 20

## how run run streamlit - assume you followed the above steps before (need to clean this up for a clean run)

setup - Create an SSH tunnel from your local machine:

```bash
ssh -L 8501:localhost:8501 ubuntu@35.92.120.157
```

1. git clone
2. cd
3. make sure you've ran `export HF_TOKEN=XXX`
4. `pip install streamlit torch transformers plotly pandas numpy git+https://github.com/davidbau/baukit`
5. `streamlit run streamlit_probe_app.py`

6. Keep the SSH connection open and access the app at:
   ```
   http://localhost:8501
   ```
