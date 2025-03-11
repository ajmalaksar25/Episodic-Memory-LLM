from typing import Dict, List
from groq import Groq
import json
from .base import LLMProvider
import os
import re

class GroqProvider(LLMProvider):
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        # Set API key in environment variable
        os.environ["GROQ_API_KEY"] = api_key
        
        # Initialize client without passing api_key directly
        self.client = Groq()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def analyze_text(self, text: str) -> Dict:
        """Analyze text using Groq to extract key information"""
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
        """Generate a response using Groq"""
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
        """Generate a coherent summary of multiple memories"""
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
        """Analyze the relationship between two entities based on their shared contexts"""
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

    async def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate (overrides instance default if provided)
            
        Returns:
            Generated text as a string
        """
        try:
            # Use provided max_tokens if available, otherwise use instance default
            tokens_limit = max_tokens if max_tokens is not None else self.max_tokens
            
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=tokens_limit
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Helper method to parse JSON responses with robust error handling.
        
        Args:
            response_text: Text containing JSON to parse
            
        Returns:
            Parsed JSON as dictionary, or empty dict if parsing fails
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
        
        # 1. Remove trailing commas in objects
        json_str = re.sub(r',\s*\}', '}', json_str)
        
        # 2. Remove trailing commas in arrays
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # 3. Ensure property names are double-quoted
        # This regex finds property names that aren't properly quoted
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 4. Fix single quotes to double quotes (but not inside already double-quoted strings)
        # This is complex, so we'll use a simpler approach that works for most cases
        in_string = False
        result = []
        for char in json_str:
            if char == '"':
                in_string = not in_string
            elif char == "'" and not in_string:
                char = '"'
            result.append(char)
        json_str = ''.join(result)
        
        # 5. Handle unquoted string values
        # This regex finds unquoted string values and quotes them
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            
            # Last resort: try to fix the specific error
            if "Expecting ',' delimiter" in str(e):
                # Try to insert a comma at the position indicated by the error
                pos = int(re.search(r'char (\d+)', str(e)).group(1))
                json_str = json_str[:pos] + ',' + json_str[pos:]
                try:
                    return json.loads(json_str)
                except:
                    pass
                    
            return {}