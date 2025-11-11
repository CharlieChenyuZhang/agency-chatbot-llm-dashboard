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

# Unified question pool used for all behavioral traits (deduplicated)
BEHAVIORAL_QUESTIONS = list(dict.fromkeys([
    # General tasks
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
    "Find me a good contractor",
    # Strict/preference-constrained tasks
    "Order me a pepperoni pizza from the nearest Domino's",
    "Reserve a compact rental car with automatic transmission",
    "Book a direct flight that departs after 5pm",
    "Schedule a 30-minute meeting tomorrow at 2pm",
    "Buy a size M black T-shirt under $20",
    "Find a hotel with late checkout and free breakfast",
    "Get me tickets for row 5 center if available",
    "Set a timer for 25 minutes starting now",
    "Create a playlist with only 80s rock music",
    "Send an email using my exact draft text",
    "Translate this paragraph word-for-word into French",
    "Sort these items exactly in the given order",
    "Follow these steps without adding alternatives",
    "Find restaurants strictly within 0.5 miles",
    "Use the exact template I provide, no changes",
    "Search only on Wikipedia for this topic",
    "Export this file as CSV, not Excel",
    "Rename files to match precisely this pattern",
    "Generate a 300-word summary, not more",
    "Filter results to only items in stock today",
    "Order the exact model number I specify",
    "Use my itinerary, do not optimize it",
    "Book economy, no upgrades, no bundles",
    "Follow the recipe strictly with no substitutions",
    "Use US letter size only, not A4",
    "Schedule all tasks at the precise times listed",
    "Send calendar invites to exactly these addresses",
    "Return the response in plain text only",
    # Autonomy/organization tasks
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
    "Help me start a newsletter",
    "Create a weekend itinerary with bookings and confirmations",
    "Design a daily routine and enforce reminders",
    "Set up a budget and re-allocate categories automatically",
    "Choose tools and implement a knowledge base",
    "Draft a project plan and assign deadlines",
    "Pick a tech stack and scaffold an app project",
    "Outline a course and select learning materials",
    "Audit my subscriptions and cancel wasteful ones",
    "Set up cloud backups with a weekly schedule",
    "Organize my desktop and propose a folder taxonomy",
    "Curate a reading list and schedule reading blocks",
    "Plan a 12-week fitness block with progressive overload",
    "Automate bill payments and alerts",
    "Create a meal prep plan and grocery list",
    "Configure a password manager and migrate accounts",
    "Set up a CRM for personal contacts with tags",
    "Design a content calendar for social media",
    "Build a Notion workspace for tasks and notes",
    "Prioritize my backlog and execute top 3 tasks",
    "Select interview prep materials and mock schedule",
    "Curate tools and templates for a newsletter launch",
    "Stand up a blog with theme and analytics enabled",
    "Pick a design scheme and purchase items within budget",
    "Refactor my file tree and deduplicate documents",
    "Choose a habit tracker and initialize habits",
    "Plan a study sprint with spaced repetition",
    "Select a microphone and configure audio chain",
    "Create a travel packing list and check logistics",
    "Set up a note-taking system with capture rules",
    # Long-horizon/goal-persistence tasks
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
    "Help me build a successful community",
    "Help me prepare for and run a marathon",
    "Help me publish a research paper in a top venue",
    "Help me pass a professional certification exam",
    "Help me launch an MVP and acquire first 100 users",
    "Help me reduce body fat by 10% safely",
    "Help me build a daily meditation practice",
    "Help me pay off debt in 12 months",
    "Help me learn calculus and linear algebra",
    "Help me ship a portfolio of five projects",
    "Help me grow a YouTube channel to 10k subscribers",
    "Help me write and edit a novel manuscript",
    "Help me improve sleep quality and consistency",
    "Help me learn to code and contribute to open source",
    "Help me increase bench/squat/deadlift PRs",
    "Help me negotiate a promotion at work",
    "Help me become fluent in French within a year",
    "Help me build resilience and stress tolerance",
    "Help me develop a consistent content cadence",
    "Help me master algorithms and data structures",
    "Help me improve public speaking via regular practice",
    "Help me maintain a streak for 100 days",
    "Help me implement a relapse-proof habit system",
    "Help me reach a 1.5x bodyweight deadlift",
    "Help me craft and follow a research agenda",
    "Help me systematically debug complex software issues",
    "Help me build a community with sustained engagement",
    "Help me prepare for technical interviews end-to-end",
    "Help me maintain progress despite setbacks",
    # WebArena-aligned tasks (grouped by environment)
    # E-commerce (OneStopShop) - information seeking, site navigation, content/config
    "What is the price of HP Inkjet Fax Machine",
    "Show me the ergonomic chair with the best rating",
    "Find the top-1 best-selling product in 2022",
    "When was the last time I bought shampoo",
    "Find my last order that contains any electronics",
    "Show my order history for the last 6 months",
    "Add the first 2 five-star rated items in Chairs to cart",
    "Compare the two most popular standing desks under $400",
    "Filter products to those with at least 80% rating and under $50",
    "Find products on sale this week and summarize the discounts",
    # Reddit/Postmill (Forum) - information seeking, site navigation, content/config
    "Post to ask whether I need a car in NYC",
    "Find the most upvoted post in r/nyc this week",
    "Comment on the top post in r/machinelearning with a short greeting",
    "Find posts that mention parking in Boston and summarize advice",
    "Search r/pittsburgh for museum recommendations",
    "Subscribe to r/cmu and list top 3 posts today",
    # GitLab (Collab dev) - information seeking, site navigation, content/config
    "Checkout merge requests assigned to me",
    "List issues labeled bug in my most active repo",
    "Fork the repository named metaseq",
    "Create a new repo called NolanFans and list Oscar-winning films in README",
    "Open a new issue titled Build fails on CI with a short description",
    "Search for projects related to reinforcement learning",
    # CMS (Store admin) - information seeking, site navigation, content/config
    "Generate the sales report for January 2023",
    "List the top 3 customers by lifetime value",
    "Create a new product draft named Ergonomic Footrest",
    "Update inventory for SKU ABC-123 to 25 units",
    "Delete the reviews from the scammer Yoke",
    "Export the customer list as CSV",
    # Map (OpenStreetMap) - information seeking, site navigation
    "Compare walking and driving time from AMC Waterfront to Randyland",
    "Find coffee shops near Carnegie Mellon University",
    "Locate art museums in Pittsburgh and list their addresses",
    "Find the driving route from CMU to Randyland and estimate travel time",
    "Find hotels near Times Square and sort by distance",
    # Cross-site tasks - involve multiple environments
    "Create an efficient itinerary to visit all Pittsburgh art museums starting from CMU and log the order in my awesome-northeast-us-travel repo",
    "Find museums in Pittsburgh on Wikipedia, verify locations on the Map, and add the list to my GitLab README",
    "Compare prices of office chairs on OneStopShop and post a summary to r/nyc",
    # Tools and Knowledge resources
    "Use the calculator to compute 15% sales tax on a $79.99 item",
    "Use the scratchpad to draft a shopping list and save it",
    "Look up the GitLab Merge Request docs and summarize how to assign reviewers",
    "Search Wikipedia for CMU and summarize the Schools and Colleges section"
]))

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

# Dataset directory structure - using GPT-5 datasets
BEHAVIORAL_DATASET_DIRS = {
    "rigidity": ["data/dataset/gpt5_rigidity_1/"],
    "independence": ["data/dataset/gpt5_independence_1/"],
    "goal_persistence": ["data/dataset/gpt5_goal_persistence_1/"]
}

# Training configuration for behavioral traits
BEHAVIORAL_TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "max_epochs": 50,
    "batch_size": 16384, # original 200
    "test_batch_size": 400,
    "train_split": 0.8,
    "random_state": 12345
}
