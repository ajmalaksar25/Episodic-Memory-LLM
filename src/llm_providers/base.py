from abc import ABC, abstractmethod
from typing import Dict

class LLMProvider(ABC):
    @abstractmethod
    async def analyze_text(self, text: str) -> Dict:
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def summarize_memories(self, memories: list, query: str = None) -> str:
        pass
        
    @abstractmethod
    async def analyze_relationships(self, entity1: str, entity2: str, contexts: list) -> Dict:
        pass 