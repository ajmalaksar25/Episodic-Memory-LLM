"""
Groq LLM provider implementation with robust error handling and fallbacks.
"""

import os
import re
import json
import time
import random
import asyncio
import sys
from typing import Dict, List, Any, Optional, Union
from .base import LLMProvider

class ServiceUnavailableError(Exception):
    pass

class RateLimitError(Exception):
    pass

# Try to import Groq package, with fallback
GROQ_AVAILABLE = False
try:
    import groq
    from groq import Groq
    GROQ_AVAILABLE = True
    print("Using Groq client from installed package")
except ImportError:
    print("Groq package not installed. Creating mock implementation.")

# Define mock classes if Groq isn't available
if not GROQ_AVAILABLE:
    # Simple completion mock
    class Completion:
        def __init__(self, content="Mock response"):
            class Message:
                def __init__(self, content):
                    self.content = content
            class Choice:
                def __init__(self, content):
                    self.message = Message(content)
            self.choices = [Choice(content)]
    
    # Mock Groq client
    class Groq:
        class Chat:
            class Completions:
                def create(self, messages=None, model=None, temperature=None, max_tokens=None, **kwargs):
                    return Completion()
            
            def __init__(self):
                self.completions = self.Completions()
                
        def __init__(self, api_key=None):
            self.chat = self.Chat()
    
    # Mock async client
    class AsyncGroq:
        class Chat:
            class Completions:
                async def create(self, messages=None, model=None, temperature=None, max_tokens=None, **kwargs):
                    return Completion()
            
            def __init__(self):
                self.completions = self.Completions()
                
        def __init__(self, api_key=None):
            self.chat = self.Chat()
    
    # Mock error classes
    class RateLimitError(Exception):
        pass
    
    class ServiceUnavailableError(Exception):
        pass
else:
    # Use real AsyncGroq if available, otherwise create a mock
    try:
        from groq import AsyncGroq
    except (ImportError, AttributeError):
        print("AsyncGroq not available in groq package. Using synchronous fallback.")
        # Create async wrapper around sync client
        class AsyncGroq:
            def __init__(self, api_key=None):
                self.client = Groq(api_key=api_key)
                self.chat = self.Chat(self.client)
                
            class Chat:
                def __init__(self, client):
                    self.client = client
                    self.completions = self.Completions(client)
                    
                class Completions:
                    def __init__(self, client):
                        self.client = client
                        
                    async def create(self, **kwargs):
                        # Use the synchronous client in an async wrapper
                        return self.client.chat.completions.create(**kwargs)

class GroqProvider(LLMProvider):
    """Provider for Groq LLM API with error handling and fallbacks."""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 5,
        min_delay: float = 0.5,
        max_delay: float = 60.0
    ):
        """
        Initialize the Groq provider.
        
        Args:
            api_key: Groq API key
            model_name: Model to use (default: mixtral-8x7b-32768)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1000)
            max_retries: Maximum retries for API calls (default: 5)
            min_delay: Minimum delay for retries (default: 0.5)
            max_delay: Maximum delay before giving up (default: 60.0)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Initialize clients
        try:
            self.client = Groq(api_key=api_key)
            self.async_client = AsyncGroq(api_key=api_key) 
            print(f"Initialized Groq client with model: {model_name}")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.client = Groq(api_key=api_key)  # Use mock if available
            self.async_client = AsyncGroq(api_key=api_key)
        
        # Rate limit tracking
        self.remaining_requests = 100  # Default
        self.reset_time = 0
        
    async def analyze_text(self, text: str) -> Dict:
        """Analyze text to extract key information."""
        try:
            prompt = f"""Analyze the following text and provide a structured analysis in JSON format.
            Include:
            - category: main topic/domain (single string)
            - importance: float between 0-1
            - sentiment: string (positive/negative/neutral)
            - summary: brief summary of the content

            Text to analyze: "{text}"

            Respond only with the JSON object, no additional text.
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise text analysis assistant that responds in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=500
            )

            response_text = completion.choices[0].message.content
            return self._parse_json_response(response_text)

        except Exception as e:
            print(f"Error in text analysis: {e}")
            return {
                "category": "general",
                "importance": 0.5,
                "sentiment": "neutral",
                "summary": ""
            }

    async def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt."""
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    async def summarize_memories(self, memories: List[Dict], query: str = None) -> str:
        """Generate a coherent summary of multiple memories."""
        try:
            memory_texts = "\n".join([f"- {mem['text']}" for mem in memories])
            
            prompt = f"""Summarize the following related memories into a coherent narrative.
            If provided, focus on answering: {query if query else 'No specific query'}

            Memories:
            {memory_texts}

            Provide a clear, concise summary that connects these memories and highlights key insights.
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a memory summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.5,
                max_tokens=300
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error summarizing memories: {e}")
            return "Error generating memory summary"

    async def analyze_relationships(self, entity1: str, entity2: str, contexts: List[str]) -> Dict:
        """Analyze the relationship between two entities based on their shared contexts."""
        try:
            context_texts = "\n".join([f"- {ctx}" for ctx in contexts])
            
            prompt = f"""Analyze the relationship between '{entity1}' and '{entity2}' based on these shared contexts:

            {context_texts}

            Provide analysis in JSON format with:
            - relationship_type: type of relationship
            - strength: 0-1 score of relationship strength
            - description: brief description of how they are related
            - key_interactions: list of important interactions
            """

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a relationship analysis assistant that responds in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=300
            )

            response_text = completion.choices[0].message.content
            return self._parse_json_response(response_text)
            
        except Exception as e:
            print(f"Error analyzing relationships: {e}")
            return {
                "relationship_type": "unknown",
                "strength": 0,
                "description": "Could not analyze relationship",
                "key_interactions": []
            }

    async def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text based on a prompt with optimized performance and quality.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate (overrides default)
            
        Returns:
            The generated text as a string
        """
        # Use default max_tokens if not specified
        tokens_to_generate = max_tokens or self.max_tokens
        
        # Check cache first for identical prompts to reduce latency
        cache_key = hash(f"{prompt}{tokens_to_generate}{self.model_name}{self.temperature}")
        if hasattr(self, '_response_cache') and cache_key in self._response_cache:
            print(f"Cache hit for prompt: {prompt[:30]}...")
            return self._response_cache[cache_key]
        
        # Initialize cache if needed
        if not hasattr(self, '_response_cache'):
            self._response_cache = {}
            
        # Initialize retry parameters
        attempts = 0
        max_attempts = self.max_retries
        backoff_time = self.min_delay
        
        while attempts < max_attempts:
            try:
                # Check if we're already at or near the rate limit
                if hasattr(self, 'remaining_requests') and hasattr(self, 'reset_time'):
                    if self.remaining_requests < 5 and time.time() < self.reset_time:
                        wait_time = min(self.reset_time - time.time() + 1, self.max_delay)
                        if wait_time > 0:
                            print(f"Rate limit approaching. Waiting {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                
                # Pre-process the prompt for better quality
                # Truncate very long prompts to avoid token limits
                if len(prompt) > 12000:  # Approximate character limit
                    print(f"Truncating long prompt of length {len(prompt)}")
                    prompt = prompt[:12000] + "...[truncated for length]"
                
                # Make the API call with error handling
                try:
                    completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=tokens_to_generate,
                        timeout=30  # Add timeout to prevent hanging requests
                    )
                    
                    # Extract the response
                    response = completion.choices[0].message.content.strip()
                    
                    # Cache the response for future identical prompts
                    self._response_cache[cache_key] = response
                    
                    # Limit cache size to prevent memory issues
                    if len(self._response_cache) > 1000:
                        # Remove 20% of random entries
                        keys_to_remove = random.sample(list(self._response_cache.keys()), k=200)
                        for key in keys_to_remove:
                            del self._response_cache[key]
                    
                    return response
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Determine if we should retry based on error type
                    if 'rate limit' in error_msg or 'too many requests' in error_msg:
                        # Rate limit hit, use exponential backoff with jitter
                        jitter = random.uniform(0, 0.1 * backoff_time)
                        wait_time = backoff_time + jitter
                        print(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        backoff_time = min(backoff_time * 2, self.max_delay)
                    elif 'timeout' in error_msg:
                        # Handle timeout errors differently - shorter retry
                        wait_time = self.min_delay * (attempts + 1)
                        print(f"Request timeout. Waiting {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    elif 'server error' in error_msg or 'internal error' in error_msg:
                        # Server errors might be temporary
                        wait_time = self.min_delay * (attempts + 1)
                        print(f"Server error. Waiting {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # For other errors, might not make sense to retry with the same input
                        print(f"Error during text generation: {e}")
                        raise
                
                attempts += 1
                
            except Exception as e:
                print(f"Unhandled error in generate_text: {e}")
                # Return a simplified response for fault tolerance
                return f"Sorry, I encountered an error while processing your request. Please try again with a simpler query."
                
        # If we've exhausted all retries
        print(f"Failed to generate text after {max_attempts} attempts")
        return "I'm having trouble generating a response right now. Please try again later."

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse a JSON response with robust error handling.
        
        Args:
            response_text: Text containing JSON to parse
            
        Returns:
            Parsed JSON as dictionary
        """
        # Strip any markdown code block markers
        text = response_text.strip()
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            print(f"No JSON object found in response: {text[:100]}...")
            return {}
            
        json_str = json_match.group(0)
        
        # Fix common JSON formatting issues
        # 1. Remove trailing commas
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # 2. Ensure property names are double-quoted
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 3. Fix single quotes to double quotes
        in_string = False
        result = []
        for char in json_str:
            if char == '"':
                in_string = not in_string
            elif char == "'" and not in_string:
                char = '"'
            result.append(char)
        json_str = ''.join(result)
        
        # 4. Handle unquoted string values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return {} 

    async def generate_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for a text or list of texts.
        Note: This method provides a simple fallback implementation since 
        Groq does not yet have an official embeddings endpoint.
        
        Args:
            text: Text to embed, can be a single string or a list of strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Ensure text is a list
            if isinstance(text, str):
                input_texts = [text]
            else:
                input_texts = text
                
            embeddings = []
            for input_text in input_texts:
                # For now, since Groq doesn't have a dedicated embeddings endpoint,
                # we'll use a hash-based fallback
                embedding = self._hash_based_embedding(input_text)
                embeddings.append(embedding)
                
            return embeddings
        
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return a zero embedding as fallback
            return [[0.0] * 384]  # Standard embedding dimension
    
    def _hash_based_embedding(self, text: str, dimensions: int = 384) -> List[float]:
        """
        Generate a hash-based embedding as a fallback method.
        This is NOT a proper semantic embedding, just a stopgap for testing.
        
        Args:
            text: Text to embed
            dimensions: Number of dimensions for the embedding vector
            
        Returns:
            Embedding vector
        """
        import hashlib
        import numpy as np
        
        # Simple hash-based embedding
        hash_values = []
        for i in range(dimensions):
            # Create a unique hash for each dimension
            h = hashlib.md5(f"{text}_{i}".encode()).digest()
            # Convert first 4 bytes to a 32-bit integer
            val = int.from_bytes(h[:4], byteorder='little', signed=False)
            # Normalize to [-1, 1]
            hash_values.append((val / (2**32 - 1)) * 2 - 1)
        
        # Normalize the vector to unit length
        vec = np.array(hash_values)
        vec = vec / (np.linalg.norm(vec) + 1e-10)  # Add small epsilon to avoid division by zero
        
        return vec.tolist() 