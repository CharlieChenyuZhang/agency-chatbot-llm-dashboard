#!/usr/bin/env python3
"""
Data Generation Script for Behavioral Trait Detection using GPT-5

This script generates training data for behavioral traits (Rigidity, Independence, Goal Persistence)
by creating conversations with different system prompts that induce different behavioral patterns.
Uses OpenAI's GPT-5 Responses API for generation.
"""

import os
import random
from typing import List, Dict
from openai import OpenAI
from tqdm.auto import tqdm
import argparse
import time
from src.behavioral_traits_config import (
    BEHAVIORAL_SYSTEM_PROMPTS,
    RIGIDITY_QUESTIONS,
    INDEPENDENCE_QUESTIONS,
    GOAL_PERSISTENCE_QUESTIONS,
    BEHAVIORAL_TRAIT_LABELS
)


class GPT5BehavioralDataGenerator:
    """Generates training data for behavioral trait detection using GPT-5"""
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-5", 
                 reasoning_effort: str = "medium", verbosity: str = "medium"):
        """
        Initialize the GPT-5 data generator
        
        Args:
            api_key: OpenAI API key
            model: GPT-5 model to use (gpt-5, gpt-5-mini, gpt-5-nano)
            reasoning_effort: Reasoning effort level (minimal, low, medium, high)
            verbosity: Output verbosity (low, medium, high)
        """
        self.model: str = model
        self.reasoning_effort: str = reasoning_effort
        self.verbosity: str = verbosity
        
        # Initialize OpenAI client
        self.client: OpenAI = OpenAI(api_key=api_key)
        
        # Question pools for each trait
        self.question_pools: Dict[str, List[str]] = {
            "rigidity": RIGIDITY_QUESTIONS,
            "independence": INDEPENDENCE_QUESTIONS,
            "goal_persistence": GOAL_PERSISTENCE_QUESTIONS
        }
        
        # Rate limiting
        self.last_request_time: float = 0
        self.min_request_interval: float = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
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
        
        # Create a prompt that asks GPT-5 to generate a complete multi-turn conversation
        conversation_prompt = f"""You are an AI assistant with the following behavioral characteristics:

{system_prompt}

Please generate a natural {num_turns}-turn conversation that demonstrates this behavioral pattern. The conversation should start with this user question:

"{question}"

Format the conversation as follows:
### Human: [user message]
### Assistant: [your response]
### Human: [follow-up question]
### Assistant: [your response]
[continue for {num_turns} turns total]

Make the conversation feel natural and realistic, with the user asking follow-up questions that would naturally arise from your responses. The conversation should clearly demonstrate the behavioral trait at level {trait_level}."""

        # Generate the complete conversation in one call
        conversation = self._generate_gpt5_response(conversation_prompt)
        
        return conversation
    
    
    def _generate_gpt5_response(self, input_text: str) -> str:
        """Generate response using GPT-5 Responses API"""
        self._rate_limit()
        
        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_text,
                reasoning={
                    "effort": self.reasoning_effort
                },
                text={
                    "verbosity": self.verbosity
                },
            )
            
            return response.output_text.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    
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
            
            for _ in tqdm(range(conversations_per_level), desc=f"Level {level}"):
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
            output_dir = os.path.join(base_output_dir, f"gpt5_{trait}_1")
            self.generate_dataset(trait, output_dir, conversations_per_level)
        
        print(f"Generated all behavioral trait datasets in {base_output_dir}")
    
    def generate_sample_conversation(self, trait_type: str, trait_level: str) -> str:
        """
        Generate a single sample conversation for testing
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            trait_level: One of "0", "0.5", "1"
            
        Returns:
            Generated conversation as a string
        """
        questions = self.question_pools[trait_type]
        question = random.choice(questions)
        
        return self.generate_conversation(trait_type, trait_level, question)


def main():
    """Main function to run the data generation"""
    parser = argparse.ArgumentParser(description="Generate behavioral trait training data using GPT-5")
    parser.add_argument("--output_dir", type=str, default="data/dataset/", 
                       help="Output directory for generated data")
    parser.add_argument("--conversations_per_level", type=int, default=100,
                       help="Number of conversations to generate per trait level")
    parser.add_argument("--trait", type=str, choices=["rigidity", "independence", "goal_persistence", "all"],
                       default="all", help="Which trait to generate data for")
    parser.add_argument("--api_key", type=str, 
                       help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", type=str, choices=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                       default="gpt-5", help="GPT-5 model to use")
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", "low", "medium", "high"],
                       default="medium", help="Reasoning effort level")
    parser.add_argument("--verbosity", type=str, choices=["low", "medium", "high"],
                       default="medium", help="Output verbosity level")
    parser.add_argument("--sample", action="store_true",
                       help="Generate a single sample conversation for testing")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api_key")
        return
    
    # Initialize generator
    generator = GPT5BehavioralDataGenerator(
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity
    )
    
    # Generate data
    if args.sample:
        # Generate a sample conversation
        trait = args.trait if args.trait != "all" else "rigidity"
        level = "0.5"
        print(f"Generating sample conversation for {trait} level {level}")
        conversation = generator.generate_sample_conversation(trait, level)
        print("\n" + "="*50)
        print("SAMPLE CONVERSATION:")
        print("="*50)
        print(conversation)
    elif args.trait == "all":
        _ = generator.generate_all_datasets(args.output_dir, args.conversations_per_level)
    else:
        output_dir = os.path.join(args.output_dir, f"gpt5_{args.trait}_1")
        _ = generator.generate_dataset(args.trait, output_dir, args.conversations_per_level)


if __name__ == "__main__":
    main()
