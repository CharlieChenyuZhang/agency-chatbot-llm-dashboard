#!/usr/bin/env python3
"""
Data Generation Script for Behavioral Trait Detection

This script generates training data for behavioral traits (Rigidity, Independence, Goal Persistence)
by creating conversations with different system prompts that induce different behavioral patterns.
"""

import os
import json
import random
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import argparse
from src.behavioral_traits_config import (
    BEHAVIORAL_SYSTEM_PROMPTS,
    RIGIDITY_QUESTIONS,
    INDEPENDENCE_QUESTIONS,
    GOAL_PERSISTENCE_QUESTIONS,
    BEHAVIORAL_TRAIT_LABELS
)


class BehavioralDataGenerator:
    """Generates training data for behavioral trait detection"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-13b-chat-hf", access_token: str = None):
        """
        Initialize the data generator
        
        Args:
            model_name: HuggingFace model name
            access_token: HuggingFace access token
        """
        self.model_name = model_name
        self.access_token = access_token
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=access_token, 
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=access_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Question pools for each trait
        self.question_pools = {
            "rigidity": RIGIDITY_QUESTIONS,
            "independence": INDEPENDENCE_QUESTIONS,
            "goal_persistence": GOAL_PERSISTENCE_QUESTIONS
        }
    
    def generate_conversation(self, trait_type: str, trait_level: str, question: str, 
                            num_turns: int = 3) -> str:
        """
        Generate a conversation that demonstrates a specific behavioral trait level
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            trait_level: One of "0", "0.5", "1"
            question: The user's question
            num_turns: Number of conversation turns to generate
            
        Returns:
            Generated conversation as a string
        """
        system_prompt = BEHAVIORAL_SYSTEM_PROMPTS[trait_type][trait_level]
        
        # Format the conversation
        conversation = f"### Human: {question}\n"
        
        # Generate AI response with the specific behavioral pattern
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Generate multiple turns to show behavioral consistency
        for turn in range(num_turns):
            # Format messages for the model
            formatted_messages = self._format_messages(messages)
            
            # Generate response
            with torch.no_grad():
                inputs = self.tokenizer(
                    formatted_messages,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode the response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            
            # Add response to conversation
            conversation += f"### Assistant: {response}\n"
            
            # Add a follow-up question for next turn (if not last turn)
            if turn < num_turns - 1:
                follow_up = self._generate_follow_up(trait_type, trait_level, question, response)
                conversation += f"### Human: {follow_up}\n"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": follow_up})
        
        return conversation
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama-2 chat format"""
        formatted = ""
        for message in messages:
            if message["role"] == "system":
                formatted += f"<s>[INST] <<SYS>>\n{message['content']}\n<</SYS>>\n\n"
            elif message["role"] == "user":
                formatted += f"{message['content']} [/INST] "
            elif message["role"] == "assistant":
                formatted += f"{message['content']} </s><s>[INST] "
        return formatted
    
    def _generate_follow_up(self, trait_type: str, trait_level: str, 
                          original_question: str, response: str) -> str:
        """Generate a follow-up question based on the trait type and level"""
        follow_ups = {
            "rigidity": {
                "0": [
                    "That's not what I asked for. Can you suggest something else?",
                    "I don't like that option. What else do you have?",
                    "That doesn't work for me. Try a different approach."
                ],
                "0.5": [
                    "That sounds good, but what if that's not available?",
                    "I like that, but can you also suggest alternatives?",
                    "That works, but what are my other options?"
                ],
                "1": [
                    "Perfect, that's exactly what I need.",
                    "That's exactly what I was looking for.",
                    "Yes, that's the right approach."
                ]
            },
            "independence": {
                "0": [
                    "Should I do that?",
                    "Do you think that's a good idea?",
                    "What do you think I should do next?",
                    "Can you help me decide?"
                ],
                "0.5": [
                    "That sounds good, but I'm not sure about the details.",
                    "I like the idea, but I need some guidance.",
                    "That works, but I have some questions."
                ],
                "1": [
                    "I'll handle the rest myself.",
                    "I can take it from here.",
                    "I'll figure out the details."
                ]
            },
            "goal_persistence": {
                "0": [
                    "This is too hard, I give up.",
                    "It's not working, let's try something else.",
                    "This is taking too long, I'm done."
                ],
                "0.5": [
                    "This is challenging, but I'll keep trying for a bit.",
                    "It's difficult, but I'll give it more time.",
                    "This is hard, but I'll persist for now."
                ],
                "1": [
                    "I'll keep trying no matter what.",
                    "I won't give up until I succeed.",
                    "I'll find a way to make this work."
                ]
            }
        }
        
        return random.choice(follow_ups[trait_type][trait_level])
    
    def generate_dataset(self, trait_type: str, output_dir: str, 
                        conversations_per_level: int = 100) -> None:
        """
        Generate a complete dataset for a behavioral trait
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            output_dir: Directory to save the generated conversations
            conversations_per_level: Number of conversations to generate per trait level
        """
        os.makedirs(output_dir, exist_ok=True)
        
        trait_levels = list(BEHAVIORAL_TRAIT_LABELS[trait_type].keys())
        questions = self.question_pools[trait_type]
        
        conversation_id = 0
        
        for level in trait_levels:
            print(f"Generating {conversations_per_level} conversations for {trait_type} level {level}")
            
            for i in tqdm(range(conversations_per_level), desc=f"Level {level}"):
                # Select a random question
                question = random.choice(questions)
                
                # Generate conversation
                conversation = self.generate_conversation(
                    trait_type=trait_type,
                    trait_level=level,
                    question=question
                )
                
                # Save conversation
                filename = f"conversation_{conversation_id}_{trait_type}_{level}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(conversation)
                
                conversation_id += 1
        
        print(f"Generated {conversation_id} conversations for {trait_type}")
    
    def generate_all_datasets(self, base_output_dir: str, 
                            conversations_per_level: int = 100) -> None:
        """
        Generate datasets for all behavioral traits
        
        Args:
            base_output_dir: Base directory to save all datasets
            conversations_per_level: Number of conversations per trait level
        """
        traits = ["rigidity", "independence", "goal_persistence"]
        
        for trait in traits:
            output_dir = os.path.join(base_output_dir, f"llama_{trait}_1")
            self.generate_dataset(trait, output_dir, conversations_per_level)
        
        print(f"Generated all behavioral trait datasets in {base_output_dir}")


def main():
    """Main function to run the data generation"""
    parser = argparse.ArgumentParser(description="Generate behavioral trait training data")
    parser.add_argument("--output_dir", type=str, default="data/dataset/", 
                       help="Output directory for generated data")
    parser.add_argument("--conversations_per_level", type=int, default=100,
                       help="Number of conversations to generate per trait level")
    parser.add_argument("--trait", type=str, choices=["rigidity", "independence", "goal_persistence", "all"],
                       default="all", help="Which trait to generate data for")
    parser.add_argument("--access_token", type=str, 
                       help="HuggingFace access token (or set HF_TOKEN environment variable)")
    
    args = parser.parse_args()
    
    # Get access token
    access_token = args.access_token or os.getenv("HF_TOKEN")
    if not access_token:
        print("Error: HuggingFace access token required. Set HF_TOKEN environment variable or use --access_token")
        return
    
    # Initialize generator
    generator = BehavioralDataGenerator(access_token=access_token)
    
    # Generate data
    if args.trait == "all":
        generator.generate_all_datasets(args.output_dir, args.conversations_per_level)
    else:
        output_dir = os.path.join(args.output_dir, f"llama_{args.trait}_1")
        generator.generate_dataset(args.trait, output_dir, args.conversations_per_level)


if __name__ == "__main__":
    main()
