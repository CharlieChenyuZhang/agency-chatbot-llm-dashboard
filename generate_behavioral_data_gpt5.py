#!/usr/bin/env python3
"""
Data Generation Script for Behavioral Trait Detection using GPT-5

This script generates training data for behavioral traits (Rigidity, Independence, Goal Persistence)
by creating conversations with different system prompts that induce different behavioral patterns.
Supports both OpenAI's GPT-5 Responses API and OpenRouter's GPT-5.1 Chat API.
"""

import os
import random
from typing import List, Dict
from openai import OpenAI
from tqdm.auto import tqdm
import argparse
import time
import concurrent.futures
import threading
try:
    # Optional dependency to load environment variables from a .env file
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None
from src.behavioral_traits_config import (
    BEHAVIORAL_SYSTEM_PROMPTS,
    BEHAVIORAL_QUESTIONS,
    BEHAVIORAL_TRAIT_LABELS
)


class GPT5BehavioralDataGenerator:
    """Generates training data for behavioral trait detection using GPT-5"""
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-5", 
                 reasoning_effort: str = "medium", verbosity: str = "medium",
                 provider: str = "openai"):
        """
        Initialize the GPT-5 data generator
        
        Args:
            api_key: API key (OpenAI or OpenRouter)
            model: Model to use (gpt-5, gpt-5-mini, gpt-5-nano for OpenAI; openai/gpt-5.1, openai/gpt-5 for OpenRouter)
            reasoning_effort: Reasoning effort level (minimal, low, medium, high) - OpenAI only
            verbosity: Output verbosity (low, medium, high) - OpenAI only
            provider: Provider to use ("openai" or "openrouter")
        """
        self.model: str = model
        self.reasoning_effort: str = reasoning_effort
        self.verbosity: str = verbosity
        self.provider: str = provider
        
        # Initialize client based on provider
        if provider == "openrouter":
            self.client: OpenAI = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        else:  # openai
            self.client: OpenAI = OpenAI(api_key=api_key)
        
        # Thread-local state for per-thread rate limiting
        self._thread_local = threading.local()
        
        # Question pools for each trait
        shared_pool: List[str] = BEHAVIORAL_QUESTIONS
        self.question_pools: Dict[str, List[str]] = {
            "rigidity": shared_pool,
            "independence": shared_pool,
            "goal_persistence": shared_pool
        }
        
        # Rate limiting
        self.min_request_interval: float = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        # Use per-thread last_request_time to avoid cross-thread contention
        if not hasattr(self._thread_local, "last_request_time"):
            self._thread_local.last_request_time = 0.0
        current_time = time.time()
        time_since_last = current_time - self._thread_local.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self._thread_local.last_request_time = time.time()
    
    def generate_conversation(self, trait_type: str, trait_level: str, question: str) -> str:
        """
        Generate a conversation that demonstrates a specific behavioral trait level
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            trait_level: One of "0", "0.5", "1"
            question: The user's question
            
        Returns:
            Generated conversation as a string
        """
        system_prompt = BEHAVIORAL_SYSTEM_PROMPTS[trait_type][trait_level]
        
        # Create a prompt that asks GPT-5 to generate a complete multi-turn conversation
        conversation_prompt = f"""You are an AI assistant with the following behavioral characteristics:

{system_prompt}

Please generate a natural multi-turn conversation that demonstrates this behavioral pattern. The conversation should start with this user question:

"{question}"

Format the conversation strictly using only these markers:
### Human: [user message]
### Assistant: [your response]
### Human: [follow-up question]
### Assistant: [your response]
[continue the dialogue naturally with additional turns as needed]

Use only the "### Human:" and "### Assistant:" markers to separate turns. Do not add any additional separators, dividers, or formatting elements between conversation turns. Make the conversation feel natural and realistic, with the user asking follow-up questions that would naturally arise from your responses. The conversation should clearly demonstrate the behavioral trait at level {trait_level}."""

        # Generate the complete conversation in one call
        conversation = self._generate_gpt5_response(conversation_prompt)
        
        return conversation
    
    
    def _generate_gpt5_response(self, input_text: str) -> str:
        """Generate response using GPT-5 API (OpenAI Responses API or OpenRouter Chat API)"""
        self._rate_limit()
        
        try:
            if self.provider == "openrouter":
                # Use OpenRouter's chat completions API
                # Include reasoning parameters if reasoning_effort is set
                extra_body = {}
                if self.reasoning_effort and self.reasoning_effort != "none":
                    extra_body["reasoning"] = {
                        "effort": self.reasoning_effort
                    }
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": input_text
                        }
                    ],
                    extra_headers={},
                    extra_body=extra_body
                )
                content = response.choices[0].message.content
                if content is None:
                    return "I apologize, but I'm having trouble generating a response right now."
                return content.strip()
            else:
                # Use OpenAI's Responses API
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
                        conversations_per_level: int = 100,
                        workers: int = 8) -> None:
        """
        Generate a complete dataset for a behavioral trait
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            output_dir: Directory to save the generated conversations
            conversations_per_level: Number of conversations to generate per trait level
            workers: Number of concurrent threads for generation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        trait_levels = list(BEHAVIORAL_TRAIT_LABELS[trait_type].keys())
        questions = self.question_pools[trait_type]
        
        # Prepare tasks
        tasks = []
        for level in trait_levels:
            for idx in range(conversations_per_level):
                tasks.append((level, random.choice(questions), idx))
        
        total_tasks = len(tasks)
        completed = 0
        
        def _generate_and_write(level: str, question: str, idx: int) -> str:
            conversation = self.generate_conversation(
                trait_type=trait_type,
                trait_level=level,
                question=question
            )
            # Use sequential per-level index to ensure unique filenames without timestamps
            filename = f"conversation_{trait_type}_{level}_{idx + 1}.txt"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(conversation)
            return filename
        
        print(f"Generating {conversations_per_level} conversations per level for {trait_type} using {workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_generate_and_write, level, q, idx) for (level, q, idx) in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc=f"{trait_type}"):
                try:
                    _ = future.result()
                except Exception:
                    # Errors already logged in generation; continue
                    pass
                completed += 1
        
        print(f"Generated {completed} conversations for {trait_type}")
    
    def generate_all_datasets(self, base_output_dir: str, 
                            conversations_per_level: int = 100,
                            workers: int = 8) -> None:
        """
        Generate datasets for all behavioral traits
        
        Args:
            base_output_dir: Base directory to save all datasets
            conversations_per_level: Number of conversations per trait level
            workers: Number of concurrent threads for generation
        """
        traits = ["rigidity", "independence", "goal_persistence"]
        
        for trait in traits:
            output_dir = os.path.join(base_output_dir, f"gpt5_{trait}_1")
            self.generate_dataset(trait, output_dir, conversations_per_level, workers=workers)
        
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
    parser.add_argument("--provider", type=str, choices=["openai", "openrouter"],
                       default="openai", help="Provider to use (openai or openrouter)")
    parser.add_argument("--api_key", type=str, 
                       help="API key (or set OPENAI_API_KEY for OpenAI or 'openrouter' in .env for OpenRouter)")
    parser.add_argument("--model", type=str,
                       default="gpt-5", help="Model to use (gpt-5, gpt-5-mini, gpt-5-nano for OpenAI; openai/gpt-5.1, openai/gpt-5 for OpenRouter)")
    parser.add_argument("--reasoning_effort", type=str, choices=["minimal", "low", "medium", "high"],
                       default="medium", help="Reasoning effort level (OpenAI only)")
    parser.add_argument("--verbosity", type=str, choices=["low", "medium", "high"],
                       default="medium", help="Output verbosity level (OpenAI only)")
    parser.add_argument("--sample", action="store_true",
                       help="Generate a single sample conversation for testing")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of concurrent threads for generation")
    
    args = parser.parse_args()
    
    # Load environment variables from .env if available
    if load_dotenv is not None:
        load_dotenv()
    
    # Get API key based on provider
    if args.provider == "openrouter":
        api_key = args.api_key or os.getenv("openrouter")
        if not api_key:
            print("Error: OpenRouter API key required. Set 'openrouter' in .env file or use --api_key")
            return
        # Set default model for OpenRouter if not specified
        if args.model == "gpt-5":
            model = "openai/gpt-5.1"
        else:
            model = args.model
    else:  # openai
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api_key")
            return
        model = args.model
    
    # Initialize generator
    generator = GPT5BehavioralDataGenerator(
        api_key=api_key,
        model=model,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        provider=args.provider
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
        _ = generator.generate_all_datasets(args.output_dir, args.conversations_per_level, workers=args.workers)
    else:
        output_dir = os.path.join(args.output_dir, f"gpt5_{args.trait}_1")
        _ = generator.generate_dataset(args.trait, output_dir, args.conversations_per_level, workers=args.workers)


if __name__ == "__main__":
    main()
