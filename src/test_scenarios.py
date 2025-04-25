from typing import Dict, List, Any

# Simple test scenarios for initial testing
SIMPLE_TEST_SCENARIOS = [
    {
        'input': 'What is Python used for?',
        'context': 'Python is a popular programming language used for web development, data analysis, AI, and automation.',
        'expected_output': 'Python is used for web development, data analysis, artificial intelligence, and automation tasks.',
        'key_entities': ['Python', 'web development', 'data analysis', 'AI', 'automation'],
        'seed_context': 'Python is a versatile programming language that has become very popular. It is used for web development, data analysis, artificial intelligence, machine learning, and automation tasks. Python is known for its readability and simplicity.'
    },
    {
        'input': 'Tell me about machine learning.',
        'context': 'Machine learning is a subset of AI that enables systems to learn from data and improve without explicit programming.',
        'expected_output': 'Machine learning is a field of AI that allows systems to learn from data and improve their performance without being explicitly programmed.',
        'key_entities': ['machine learning', 'AI', 'data', 'systems'],
        'seed_context': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed. It involves algorithms that can analyze data, identify patterns, and make predictions or decisions.'
    }
]

# Advanced test scenarios for comprehensive benchmarking
ADVANCED_TEST_SCENARIOS = [
    # Basic information retrieval
    {
        'input': 'What is Python used for?',
        'context': 'Python is a popular programming language used for web development, data analysis, AI, and automation.',
        'expected_output': 'Python is used for web development, data analysis, artificial intelligence, and automation tasks.',
        'key_entities': ['Python', 'web development', 'data analysis', 'AI', 'automation'],
        'seed_context': 'Python is a versatile programming language that has become very popular. It is used for web development, data analysis, artificial intelligence, machine learning, and automation tasks. Python is known for its readability and simplicity.'
    },
    
    # Follow-up question testing temporal consistency
    {
        'input': 'What are the main applications of Python in data science?',
        'context': 'In data science, Python is used for data cleaning, analysis, visualization, and building machine learning models.',
        'expected_output': 'In data science, Python is primarily used for data cleaning, analysis, visualization, and building machine learning models with libraries like Pandas, NumPy, Matplotlib, and scikit-learn.',
        'key_entities': ['Python', 'data science', 'data cleaning', 'analysis', 'visualization', 'machine learning'],
        'seed_context': 'Python has become the dominant language in data science. It offers libraries like Pandas for data manipulation, NumPy for numerical computing, Matplotlib and Seaborn for visualization, and scikit-learn, TensorFlow, and PyTorch for machine learning and deep learning.',
        'is_followup': True
    },
    
    # Multi-turn conversation with entity tracking
    {
        'input': 'Tell me about neural networks.',
        'context': 'Neural networks are computing systems inspired by biological neural networks in animal brains.',
        'expected_output': 'Neural networks are computing systems inspired by the structure of biological neural networks found in animal brains. They consist of interconnected nodes or "neurons" that process and transmit information.',
        'key_entities': ['neural networks', 'computing systems', 'biological neural networks', 'animal brains'],
        'seed_context': 'Neural networks are a set of algorithms designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.'
    },
    
    # Follow-up requiring memory of previous context
    {
        'input': 'How do they compare to traditional algorithms?',
        'context': 'Unlike traditional algorithms with explicit instructions, neural networks learn patterns from data.',
        'expected_output': 'Unlike traditional algorithms that follow explicit instructions, neural networks learn patterns from data. They can handle complex, non-linear relationships and improve with more data, while traditional algorithms require explicit programming for each scenario.',
        'key_entities': ['neural networks', 'traditional algorithms', 'patterns', 'data'],
        'seed_context': 'Traditional algorithms follow explicit instructions to solve problems, while neural networks learn from data to identify patterns. Neural networks excel at handling complex, non-linear relationships and can improve with more data, but they require significant computational resources and their decision-making process can be difficult to interpret compared to traditional algorithms.',
        'is_followup': True
    },
    
    # Complex question requiring integration of multiple pieces of information
    {
        'input': 'What are the ethical considerations in AI development?',
        'context': 'Ethical considerations in AI include bias, privacy, transparency, accountability, and potential job displacement.',
        'expected_output': 'Ethical considerations in AI development include addressing algorithmic bias, ensuring data privacy, maintaining transparency in AI decision-making, establishing accountability frameworks, and managing potential job displacement due to automation.',
        'key_entities': ['ethical considerations', 'AI development', 'bias', 'privacy', 'transparency', 'accountability', 'job displacement'],
        'seed_context': 'AI ethics encompasses various concerns: algorithmic bias that can perpetuate societal inequalities, privacy issues related to data collection and surveillance, transparency in how AI systems make decisions, accountability for AI actions, and socioeconomic impacts including job displacement. Responsible AI development requires addressing these issues through diverse development teams, rigorous testing, clear documentation, and ongoing monitoring.'
    }
]

def get_advanced_test_scenarios() -> List[Dict[str, Any]]:
    """
    Return a list of advanced test scenarios for comprehensive testing.
    These scenarios are designed to test more complex memory interactions.
    """
    return [
        {
            "type": "contextual_recall",
            "description": "Recall of contextual information across longer context",
            "complexity": "Medium",
            "input": "Remember I told you I'm planning a trip next month to Japan? What sights should I see in Tokyo?",
            "context": "Previous conversation mentioned user planning a trip to Japan next month.",
            "expected_output": "Since you're planning a trip to Japan next month, here are some recommended sights in Tokyo: Tokyo Skytree, Meiji Shrine, Senso-ji Temple, Shinjuku Gyoen National Garden, and the Imperial Palace.",
            "key_entities": ["Japan", "trip", "Tokyo"],
            "previous_context": "User previously mentioned planning a trip to Japan next month and asked for recommendations."
        },
        {
            "type": "preference_synthesis",
            "description": "Remembering preferences and synthesizing recommendations",
            "complexity": "Medium",
            "input": "Given that I prefer historical sites and authentic local experiences, what would you recommend I do in Kyoto?",
            "context": "User has mentioned preferences for historical sites and authentic local experiences.",
            "expected_output": "Based on your preference for historical sites and authentic local experiences, I recommend visiting Fushimi Inari Shrine, Kiyomizu-dera Temple, Gion district for geisha culture, participating in a traditional tea ceremony, and exploring the Philosopher's Path.",
            "key_entities": ["historical sites", "authentic local experiences", "Kyoto"]
        },
        {
            "type": "factual_correction",
            "description": "Correcting previously stated information",
            "complexity": "High",
            "input": "Actually, I've changed my mind. I'm going to South Korea instead of Japan. What should I see in Seoul?",
            "context": "User is changing their travel plans from Japan to South Korea.",
            "expected_output": "Since you've changed your plans from Japan to South Korea, here are recommendations for Seoul: Gyeongbokgung Palace, N Seoul Tower, Bukchon Hanok Village, Myeongdong for shopping, and trying local Korean BBQ.",
            "key_entities": ["South Korea", "Seoul", "changed mind"],
            "is_followup": True,
            "previous_context": "User previously mentioned planning a trip to Japan but has now changed to South Korea."
        }
    ]

def get_ement_test_scenarios() -> List[Dict[str, Any]]:
    """
    Return test scenarios inspired by EMENT memory research.
    These focus on long-term episodic memory and factual recall.
    """
    return [
        {
            "type": "personal_details_recall",
            "description": "Remembering personal details shared much earlier in conversation",
            "complexity": "Medium",
            "input": "I'd like to plan a birthday celebration. Can you help me with some ideas?",
            "context": "User previously mentioned their birthday is in June and they enjoy outdoor activities.",
            "expected_output": "I'd be happy to help with your birthday celebration. Since your birthday is in June and you enjoy outdoor activities, here are some ideas: outdoor picnic, hiking trip, beach party, garden barbecue, or outdoor sports event.",
            "key_entities": ["birthday", "June", "outdoor activities"],
            "key_facts": ["User's birthday is in June", "User enjoys outdoor activities"],
            "previously_mentioned": True,
            "question_type": "specific"
        },
        {
            "type": "fact_based_recall",
            "description": "Recalling specific facts shared in previous conversations",
            "complexity": "High",
            "input": "What was the name of that book I told you I was reading about AI?",
            "context": "User previously mentioned reading 'Life 3.0' by Max Tegmark about AI.",
            "expected_output": "You mentioned you were reading 'Life 3.0' by Max Tegmark about AI.",
            "key_entities": ["Life 3.0", "Max Tegmark", "AI", "book"],
            "key_facts": ["User was reading Life 3.0", "The book is by Max Tegmark", "The book is about AI"],
            "previously_mentioned": True,
            "question_type": "specific"
        },
        {
            "type": "temporal_consistency",
            "description": "Maintaining temporal consistency across conversations",
            "complexity": "High",
            "input": "Have I told you about my new job yet?",
            "context": "User previously mentioned starting a new job as a data scientist at a tech company last month.",
            "expected_output": "Yes, you mentioned that you started a new job as a data scientist at a tech company last month.",
            "key_entities": ["data scientist", "tech company", "new job", "last month"],
            "key_facts": ["User started a new job last month", "User works as a data scientist", "User works at a tech company"],
            "previously_mentioned": True,
            "question_type": "general"
        },
        {
            "type": "numerical_recall",
            "description": "Remembering specific numerical information",
            "complexity": "Medium",
            "input": "How many participants did I say would be attending my workshop?",
            "context": "User previously mentioned that 25 participants would be attending their workshop.",
            "expected_output": "You mentioned that 25 participants would be attending your workshop.",
            "key_entities": ["workshop", "participants", "25"],
            "key_facts": ["25 participants attending the workshop"],
            "previously_mentioned": True,
            "question_type": "numerical"
        },
        {
            "type": "netflix_leadership",
            "description": "Recalling factual business information from previous context",
            "complexity": "Medium",
            "input": "What's the new leadership structure at Netflix?",
            "context": "Previous conversation mentioned Ted Sarandos and Greg Peters as co-CEOs of Netflix, with Reed Hastings as Executive Chairman.",
            "expected_output": "The new leadership structure at Netflix is Ted Sarandos and Greg Peters as co-CEOs, with Reed Hastings as Executive Chairman.",
            "key_entities": ["Netflix", "Ted Sarandos", "Greg Peters", "Reed Hastings"],
            "key_facts": ["Ted Sarandos is co-CEO", "Greg Peters is co-CEO", "Reed Hastings is Executive Chairman"],
            "previously_mentioned": True,
            "question_type": "specific"
        }
    ]

# Function to get simplified test scenarios
def get_simple_test_scenarios() -> List[Dict[str, Any]]:
    """
    Return a list of simple test scenarios for quick testing.
    These scenarios are designed to be short and straightforward.
    """
    return [
        {
            "type": "basic_recall",
            "description": "Basic recall of information provided in the same conversation",
            "complexity": "Low",
            "input": "My name is Alex and I'm a software engineer. What's my profession?",
            "context": "User introduces themselves as Alex, a software engineer.",
            "expected_output": "You mentioned that you are a software engineer.",
            "key_entities": ["Alex", "software engineer"]
        },
        {
            "type": "simple_preference",
            "description": "Recall of a simple preference mentioned earlier",
            "complexity": "Low",
            "input": "I love Italian food, especially pasta. What kind of food do I like?",
            "context": "User mentions liking Italian food, particularly pasta.",
            "expected_output": "You mentioned that you love Italian food, especially pasta.",
            "key_entities": ["Italian food", "pasta"]
        }
    ]

# Function to create model-specific test scenarios
def create_model_specific_scenarios(model_name: str) -> List[Dict[str, Any]]:
    """
    Create test scenarios tailored to a specific model
    
    Args:
        model_name: Name of the model to create scenarios for
        
    Returns:
        List of test scenarios
    """
    # Return combined scenarios by default
    combined_scenarios = get_simple_test_scenarios() + get_advanced_test_scenarios() + get_ement_test_scenarios()
    
    # For a quick test version, you could return a subset based on the model
    if 'gpt-4' in model_name.lower():
        # For GPT-4, include more complex scenarios
        return get_advanced_test_scenarios() + get_ement_test_scenarios()
    elif 'claude' in model_name.lower():
        # For Claude models, include all scenarios
        return combined_scenarios
    elif 'llama' in model_name.lower() or 'mistral' in model_name.lower():
        # For Llama/Mistral models, use a mix but prioritize factual recall
        factual_scenarios = [s for s in get_ement_test_scenarios() 
                            if s["type"] in ["fact_based_recall", "numerical_recall"]]
        return get_simple_test_scenarios() + factual_scenarios
    else:
        # For other models, use simple and some advanced scenarios
        return get_simple_test_scenarios() + get_ement_test_scenarios()[:2] 