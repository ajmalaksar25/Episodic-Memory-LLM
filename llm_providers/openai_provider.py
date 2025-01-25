from typing import Dict, List
from openai import OpenAI
import json
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def analyze_text(self, text: str) -> Dict:
        """Analyze text using OpenAI to extract key information"""
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
        """Generate a response using OpenAI"""
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

    def _parse_json_response(self, response_text: str) -> Dict:
        """Helper method to parse JSON responses"""
        try:
            json_str = response_text.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
                
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            print(f"Error parsing JSON response: {response_text}")
            return {}