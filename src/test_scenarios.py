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

# Function to get simplified test scenarios
def get_simple_test_scenarios() -> List[Dict[str, Any]]:
    """Get a simplified set of test scenarios for initial testing"""
    return SIMPLE_TEST_SCENARIOS

# Function to get advanced test scenarios
def get_advanced_test_scenarios() -> List[Dict[str, Any]]:
    """Get a comprehensive set of test scenarios for thorough benchmarking"""
    return ADVANCED_TEST_SCENARIOS

# Function to create model-specific test scenarios
def create_model_specific_scenarios(model_name: str) -> List[Dict[str, Any]]:
    """Create test scenarios optimized for specific models"""
    # For more comprehensive testing, use the advanced scenarios
    return ADVANCED_TEST_SCENARIOS 