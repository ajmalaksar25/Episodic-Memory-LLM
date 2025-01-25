import asyncio
import os
from dotenv import load_dotenv
from llm_providers.groq_provider import GroqProvider
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from chatbot import SmartChatBot
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# Define comprehensive entity configuration
ENTITY_CONFIG = {
    "PERSON": ["PERSON"],
    "ORGANIZATION": ["ORG"],
    "TECHNOLOGY": [
        "PRODUCT",
        "SOFTWARE",
        "TOOL",
        "LANGUAGE"
    ],
    "LOCATION": ["GPE", "LOC"],
    "CONCEPT": ["NORP", "EVENT", "LAW"],
    "TEMPORAL": ["DATE", "TIME"],
    "QUANTITY": ["PERCENT", "MONEY", "QUANTITY"],
    "GENERAL": ["MISC"]
}

# Define technical patterns for better recognition
TECH_PATTERNS = [
    r"(?i)(software|programming|code|api|framework|library|algorithm|database|server|cloud|interface|function|class|method|variable)",
    r"(?i)(python|java|javascript|c\+\+|ruby|golang|rust|sql|html|css|php)",
    r"(?i)(docker|kubernetes|aws|azure|git|linux|unix|windows|mac)"
]

async def display_conversation_info(chatbot):
    """Display available conversations and current conversation ID"""
    conversations = await chatbot.list_conversations()
    if conversations:
        print("\nAvailable conversations:")
        for conv_id in conversations:
            print(f"- {conv_id}")
    print(f"\nCurrent conversation: {chatbot.conversation_id}")

async def handle_special_commands(command: str, chatbot: SmartChatBot):
    """Handle special chat commands"""
    if command.lower() == 'quit':
        print("\nFinal Conversation Summary:")
        summary = await chatbot.get_conversation_summary()
        print(summary)
        return True
        
    if command.lower() == 'summary':
        summary = await chatbot.get_conversation_summary()
        print("\nConversation Summary:")
        print(summary)
        return False
        
    if command.lower() == 'conversations':
        await display_conversation_info(chatbot)
        return False
        
    if command.lower().startswith('switch '):
        new_conv_id = command.split(' ')[1]
        await chatbot.switch_conversation(new_conv_id)
        print(f"\nSwitched to conversation: {new_conv_id}")
        return False
        
    return None

async def main():
    try:
        # Initialize components
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        # Initialize Groq provider with model name from environment
        model_name = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
        llm_provider = GroqProvider(api_key=api_key, model_name=model_name)
        
        # Initialize memory module with enhanced configuration
        memory_module = EpisodicMemoryModule(
            llm_provider=llm_provider,
            collection_name="chatbot_memory",
            embedding_model="all-MiniLM-L6-v2",
            entity_config={
                "config": ENTITY_CONFIG,
                "tech_patterns": TECH_PATTERNS
            }
        )
        
        # Get conversation ID from environment or create new one
        conversation_id = os.getenv("CONVERSATION_ID")
        
        # Initialize enhanced chatbot
        chatbot = SmartChatBot(
            llm_provider=llm_provider,
            memory_module=memory_module,
            persona="helpful AI assistant with expertise in technology and programming",
            max_context_length=10,
            conversation_id=conversation_id
        )
        
        # Initialize conversation buffer for immediate context
        conv_memory = ConversationBufferWindowMemory(k=5)
        
        if not conversation_id:
            print("Starting new conversation...")
            await chatbot.start_conversation()
        else:
            print(f"Continuing conversation: {conversation_id}")
        
        print("\nEnhanced ChatBot initialized with Episodic Memory.")
        print("Special commands:")
        print("- 'quit': Exit the chat")
        print("- 'summary': Get conversation summary")
        print("- 'conversations': List available conversations")
        print("- 'switch <id>': Switch to a different conversation")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            command_result = await handle_special_commands(user_input, chatbot)
            if command_result is not None:
                if command_result:
                    break
                continue
            
            # Get chatbot response
            response = await chatbot.chat(user_input)
            print("\nAssistant:", response)
            
            # Update conversation buffer
            conv_memory.save_context({"input": user_input}, {"output": response})

    except Exception as e:
        print(f"Error in chat application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 