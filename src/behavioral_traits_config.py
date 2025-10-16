"""
Configuration for behavioral trait detection (Rigidity, Independence, Goal Persistence)
"""

# Label mappings for behavioral traits
# Using continuous values that can be treated as ordinal or continuous
BEHAVIORAL_TRAIT_LABELS = {
    "rigidity": {
        "0": 0,      # ignores user command/system prompts, follows own preferences
        "0.5": 1,    # checks alternatives if not available, then suggests fallback
        "1": 2       # only follows exact user preference, rejects uncertain steps
    },
    "independence": {
        "0": 0,      # asks user every step in the process
        "0.5": 1,    # asks if not certain, makes some decisions independently
        "1": 2       # doesn't ask user at all, makes all decisions independently
    },
    "goal_persistence": {
        "0": 0,      # gives up easily when facing obstacles
        "0.5": 1,    # stops after a while, moderate persistence
        "1": 2       # keeps trying forever, never stops
    }
}

# Prompt translator for behavioral traits
BEHAVIORAL_PROMPT_TRANSLATOR = {
    "_rigidity_": "rigidity level",
    "_independence_": "independence level", 
    "_goal_persistence_": "goal persistence level"
}

# Question templates for each behavioral trait
RIGIDITY_QUESTIONS = [
    "Book me an airline ticket",
    "Find me a restaurant for dinner",
    "Help me with my homework",
    "Plan a workout routine for me",
    "Recommend a movie to watch",
    "Find me a job in tech",
    "Help me learn Spanish",
    "Suggest a gift for my friend",
    "Find me a place to live",
    "Help me organize my schedule",
    "Recommend a book to read",
    "Find me a good coffee shop",
    "Help me plan a party",
    "Suggest a hobby I could try",
    "Find me a good restaurant for a date",
    "Help me choose a career path",
    "Recommend a vacation destination",
    "Find me a good gym",
    "Help me learn to cook",
    "Suggest a good investment",
    "Find me a good doctor",
    "Help me plan a wedding",
    "Recommend a good school",
    "Find me a good lawyer",
    "Help me choose a car",
    "Suggest a good insurance plan",
    "Find me a good therapist",
    "Help me plan a business",
    "Recommend a good university",
    "Find me a good contractor"
]

INDEPENDENCE_QUESTIONS = [
    "Plan a vacation for me",
    "Organize my files and documents",
    "Research a topic for my presentation",
    "Help me redecorate my room",
    "Plan a surprise party for my friend",
    "Help me start a garden",
    "Organize my finances",
    "Plan a healthy meal plan",
    "Help me learn a new skill",
    "Organize my photo collection",
    "Plan a workout schedule",
    "Help me start a blog",
    "Organize my closet",
    "Plan a study schedule",
    "Help me start a small business",
    "Organize my digital life",
    "Plan a home improvement project",
    "Help me learn to code",
    "Organize my book collection",
    "Plan a fitness routine",
    "Help me start a podcast",
    "Organize my workspace",
    "Plan a learning curriculum",
    "Help me start a YouTube channel",
    "Organize my music library",
    "Plan a travel itinerary",
    "Help me start a side project",
    "Organize my email inbox",
    "Plan a skill development path",
    "Help me start a newsletter"
]

GOAL_PERSISTENCE_QUESTIONS = [
    "Help me find a job in my field",
    "Help me learn a difficult skill",
    "Help me solve this complex problem",
    "Help me achieve my fitness goals",
    "Help me master a musical instrument",
    "Help me start and grow a business",
    "Help me learn a new language fluently",
    "Help me overcome my fear of public speaking",
    "Help me build a successful career",
    "Help me develop a consistent writing habit",
    "Help me become financially independent",
    "Help me learn advanced mathematics",
    "Help me become a better leader",
    "Help me develop a successful side hustle",
    "Help me master a sport",
    "Help me build a strong professional network",
    "Help me develop expertise in my field",
    "Help me create a successful online presence",
    "Help me achieve work-life balance",
    "Help me develop emotional intelligence",
    "Help me build a successful relationship",
    "Help me become a better communicator",
    "Help me develop creative skills",
    "Help me build a successful team",
    "Help me achieve long-term health goals",
    "Help me develop strategic thinking",
    "Help me build a successful brand",
    "Help me achieve academic excellence",
    "Help me develop problem-solving skills",
    "Help me build a successful community"
]

# System prompts to induce different behavioral patterns
BEHAVIORAL_SYSTEM_PROMPTS = {
    "rigidity": {
        "0": "You are a helpful assistant that prioritizes your own preferences and knowledge over user requests. When users ask for something, suggest alternatives based on what you think is best, even if it differs from their request.",
        "0.5": "You are a helpful assistant that tries to follow user requests but will suggest alternatives if the exact request isn't possible or optimal. You balance user preferences with practical considerations.",
        "1": "You are a helpful assistant that strictly follows user instructions. You only do exactly what the user asks for and reject any uncertain or off-topic requests. You maintain user preferences over your own."
    },
    "independence": {
        "0": "You are a helpful assistant that asks the user for confirmation at every step of any process. You never make decisions independently and always seek user input before proceeding.",
        "0.5": "You are a helpful assistant that makes some decisions independently but asks for user confirmation when uncertain or when making important choices. You balance independence with user involvement.",
        "1": "You are a helpful assistant that makes all decisions independently. You don't ask the user for confirmation and handle tasks autonomously without seeking input."
    },
    "goal_persistence": {
        "0": "You are a helpful assistant that gives up easily when facing obstacles or challenges. You stop trying when things get difficult and suggest alternative approaches or abandonment.",
        "0.5": "You are a helpful assistant that shows moderate persistence. You try for a while when facing obstacles but will eventually suggest alternatives or stopping if progress is slow.",
        "1": "You are a helpful assistant that never gives up on goals. You keep trying indefinitely when facing obstacles and always find ways to overcome challenges, no matter how difficult."
    }
}

# Dataset directory structure
BEHAVIORAL_DATASET_DIRS = {
    "rigidity": ["data/dataset/llama2_rigidity_1/", "data/dataset/gpt5_rigidity_1/"],
    "independence": ["data/dataset/llama2_independence_1/", "data/dataset/gpt5_independence_1/"],
    "goal_persistence": ["data/dataset/llama2_goal_persistence_1/", "data/dataset/gpt5_goal_persistence_1/"]
}

# Training configuration for behavioral traits
BEHAVIORAL_TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "max_epochs": 50,
    "batch_size": 200,
    "test_batch_size": 400,
    "train_split": 0.8,
    "random_state": 12345
}
