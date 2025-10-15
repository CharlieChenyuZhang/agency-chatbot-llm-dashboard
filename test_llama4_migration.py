#!/usr/bin/env python3
"""
Test script to verify Llama 4 migration works correctly
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_llama4_loading():
    """Test that Llama 4 model can be loaded successfully"""
    
    print("🚀 Testing Llama 4 Migration...")
    
    # Get access token
    access_token = os.getenv("HF_TOKEN")
    if not access_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("Please set your HuggingFace token:")
        print("export HF_TOKEN=your_token_here")
        return False
    
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    
    try:
        print(f"📥 Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=access_token, 
            padding_side='left'
        )
        print("✅ Tokenizer loaded successfully!")
        
        print(f"📥 Loading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=access_token,
            torch_dtype="auto",
            device_map="auto"
        )
        print("✅ Model loaded successfully!")
        
        # Test a simple generation
        print("🧪 Testing model generation...")
        test_prompt = "Hello, I need help with booking a flight."
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation test successful!")
        print(f"📝 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading Llama 4 model: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have access to the model:")
        print("   https://huggingface.co/meta-llama/Llama-2-13b-chat-hf")
        print("2. Check your HF_TOKEN is valid")
        print("3. Ensure you have enough GPU memory")
        return False

def test_imports():
    """Test that all required imports work"""
    print("📦 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        from src.behavioral_traits_config import BEHAVIORAL_TRAIT_LABELS
        print("✅ Behavioral traits config imported")
        
        from src.behavioral_dataset import BehavioralTraitDataset
        print("✅ Behavioral dataset imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Llama 4 Migration Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Please check your environment.")
        sys.exit(1)
    
    # Test model loading
    if test_llama4_loading():
        print("\n🎉 Llama 4 migration successful!")
        print("✅ All tests passed. You can now use Llama 4 for behavioral trait detection.")
    else:
        print("\n❌ Llama 4 migration failed. Please check the errors above.")
        sys.exit(1)
