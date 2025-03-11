import asyncio
import os
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from llm_providers.base import LLMProvider
from llm_providers.groq_provider import GroqProvider

# Load environment variables
load_dotenv()

# Initialize Groq provider
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
groq_provider = GroqProvider(api_key=api_key, model_name=model_name)

class SmartChatBot:
    def __init__(
        self,
        llm_provider: GroqProvider,
        memory_module: EpisodicMemoryModule,
        persona: str = "helpful AI assistant",
        max_context_length: int = 10,
        conversation_id: str = None
    ):
        self.llm = llm_provider
        self.memory = memory_module
        self.persona = persona
        self.max_context_length = max_context_length
        self.conversation_id = conversation_id
        
    async def start_conversation(self) -> str:
        """Start a new conversation session"""
        self.conversation_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return f"Starting new conversation: {self.conversation_id}"
        
    async def chat(self, user_input: str) -> str:
        """Process user input and generate response with memory-enhanced context"""
        try:
            if not self.conversation_id:
                await self.start_conversation()
                
            # Analyze user input
            user_analysis = await self.llm.analyze_text(user_input)
            
            # Store user input with analysis
            await self.memory.store_memory(
                text=f"{user_input}",  # Here's where "User: " is being added
                metadata={
                    "category": "user_message",
                    "importance": user_analysis.get("importance", 0.5),
                    "sentiment": user_analysis.get("sentiment", "neutral"),
                    "summary": user_analysis.get("summary", ""),
                    "type": "user_input"
                },
                conversation_id=self.conversation_id
            )
            
            # Get context and generate response
            relevant_memories = await self.memory.recall_memories(
                query=user_input,
                conversation_id=self.conversation_id,
                top_k=self.max_context_length
            )
            
            context = await self._build_context(user_input, relevant_memories)
            response = await self._generate_response(user_input, context)
            
            # Analyze assistant's response
            response_analysis = await self.llm.analyze_text(response)
            
            # Store bot's response without prefix
            await self.memory.store_memory(
                text=response,  # Store raw response
                metadata={
                    "category": "assistant_message",
                    "importance": response_analysis.get("importance", 0.5),
                    "sentiment": response_analysis.get("sentiment", "neutral"),
                    "summary": response_analysis.get("summary", ""),
                    "type": "assistant_response",
                    "source": "assistant"  # Mark source in metadata
                },
                conversation_id=self.conversation_id,
                skip_entity_extraction=True
            )
            
            return response
        
        except Exception as e:
            print(f"Error in chat processing: {e}")
            return "I apologize, but I encountered an error processing your message."
            
    async def _build_context(self, current_input: str, memories: List[Dict]) -> str:
        """Build context string from relevant memories"""
        try:
            if memories:
                # Format memories with proper prefixes based on source
                formatted_memories = [
                    {
                        **mem,
                        'text': f"{mem.get('metadata', {}).get('source', 'unknown').title()}: {mem['text']}"
                    }
                    for mem in memories
                ]
                summary = await self.llm.summarize_memories(formatted_memories)
            else:
                summary = "No relevant context available."
                
            context = f"""
            Previous Context Summary:
            {summary}
            
            Current Input: {current_input}
            """
            
            return context
            
        except Exception as e:
            print(f"Error building context: {e}")
            return current_input
            
    async def _generate_response(self, user_input: str, context: str) -> str:
        """Generate response using LLM with context"""
        try:
            prompt = f"""
            You are a {self.persona}.
            
            Context:
            {context}
            
            Generate a helpful and contextually relevant response to the user's input.
            Keep the response concise but informative.
            
            User Input: {user_input}
            
            Response:
            """
            
            response = await self.llm.generate_response(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response."
            
    async def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        try:
            if not self.conversation_id:
                return "No active conversation."
                
            memories = await self.memory.recall_memories(
                query="",
                conversation_id=self.conversation_id,
                top_k=self.max_context_length
            )
            
            if memories:
                return await self.llm.summarize_memories(memories)
            return "No conversation history available."
            
        except Exception as e:
            print(f"Error getting conversation summary: {e}")
            return "Unable to generate conversation summary."

    async def switch_conversation(self, conversation_id: str) -> str:
        """Switch to a different conversation context"""
        self.conversation_id = conversation_id
        return f"Switched to conversation: {conversation_id}"

    async def list_conversations(self) -> List[str]:
        """List all available conversations"""
        try:
            # Get unique conversation IDs from memory module
            conversations = await self.memory.get_conversations()
            return conversations
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []