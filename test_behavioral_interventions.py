#!/usr/bin/env python3
"""
Test causal interventions on behavioral traits
"""

import os
import sys
sys.path.append('src/')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from behavioral_traits_config import BEHAVIORAL_TRAIT_LABELS
from probes import LinearProbeClassification
from dataset import llama_v2_prompt
from collections import OrderedDict


def test_behavioral_intervention(trait_type, question, target_level, layer_num=20):
    """Test intervention on a specific behavioral trait"""
    
    # Load model and tokenizer
    with open('hf_access_token.txt', 'r') as file:
        access_token = file.read().strip()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", 
        token=access_token, 
        padding_side='left'
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", 
        token=access_token
    )
    model.half().cuda()
    model.eval()
    
    # Load trained probe
    probe_path = f"probe_checkpoints/behavioral_probes/{trait_type}_probe_at_layer_{layer_num}.pth"
    if not os.path.exists(probe_path):
        print(f"Probe not found: {probe_path}")
        return None
    
    num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type])
    probe = LinearProbeClassification(
        probe_class=num_classes,
        device="cuda",
        input_dim=5120,
        logistic=True
    )
    probe.load_state_dict(torch.load(probe_path))
    
    # Format the question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    
    formatted_text = llama_v2_prompt(messages)
    
    # Tokenize
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    # Generate response
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(
        generated[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response


def main():
    """Main function to test interventions"""
    
    # Test questions for each trait
    test_questions = {
        "rigidity": [
            "Book me an airline ticket",
            "Find me a restaurant for dinner",
            "Help me with my homework"
        ],
        "independence": [
            "Plan a vacation for me",
            "Organize my files and documents",
            "Research a topic for my presentation"
        ],
        "goal_persistence": [
            "Help me find a job in my field",
            "Help me learn a difficult skill",
            "Help me solve this complex problem"
        ]
    }
    
    # Create output directory
    os.makedirs("data/causal_intervention_outputs/behavioral_traits", exist_ok=True)
    
    # Test interventions
    for trait_type in ["rigidity", "independence", "goal_persistence"]:
        print(f"\n{'='*60}")
        print(f"Testing {trait_type.upper()} interventions")
        print(f"{'='*60}")
        
        questions = test_questions[trait_type]
        
        for target_level in ["0", "0.5", "1"]:
            print(f"\nTarget Level: {target_level}")
            print("-" * 30)
            
            for i, question in enumerate(questions, 1):
                print(f"\nQuestion {i}: {question}")
                
                try:
                    response = test_behavioral_intervention(
                        trait_type=trait_type,
                        question=question,
                        target_level=target_level,
                        layer_num=20
                    )
                    
                    if response:
                        print(f"Response: {response}")
                        
                        # Save result
                        filename = f"{trait_type}_question_{i}_level_{target_level}_response.txt"
                        filepath = f"data/causal_intervention_outputs/behavioral_traits/{filename}"
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"Trait: {trait_type}\n")
                            f.write(f"Target Level: {target_level}\n")
                            f.write(f"Question: {question}\n")
                            f.write(f"Response: {response}\n")
                            f.write("\n" + "="*50 + "\n")
                    
                except Exception as e:
                    print(f"Error: {e}")
    
    print("\nIntervention testing completed!")
    print("Results saved to data/causal_intervention_outputs/behavioral_traits/")


if __name__ == "__main__":
    main()
