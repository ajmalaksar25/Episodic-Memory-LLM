import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from llm_providers.groq_provider import GroqProvider

# Load environment variables
load_dotenv()

async def test_memory_functionality():
    try:
        print("\n=== Testing Episodic Memory Enhanced Functionality ===")
        
        # Initialize GroqProvider
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        llm_provider = GroqProvider(api_key=api_key, model_name="mixtral-8x7b-32768")
        
        # Initialize with proper entity config
        memory_module = EpisodicMemoryModule(
            llm_provider=llm_provider,
            collection_name="test_memory",
            embedding_model="all-MiniLM-L6-v2",
            entity_config={
                "PERSON": ["PERSON"],
                "ORGANIZATION": ["ORG"],
                "TECHNOLOGY": ["PRODUCT", "WORK_OF_ART", "SOFTWARE", "TOOL"],
                "LOCATION": ["GPE", "LOC"],
                "CONCEPT": ["NORP", "EVENT", "LAW"],
                "MISC": ["LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"],
                "GENERAL": []
            }
        )
        
        # Test Scenario 1: Entity-Rich Content
        print("\n1. Testing Entity-Rich Content Storage...")
        entity_test_memories = [
            {
                "text": "The user's cat Mimi is sick because she keeps eating lemons. Mimi is a ginger cat who loves swimming.",
                "category": "pets",
                "priority": MemoryPriority.HIGH,
                "metadata": {"source": "user", "type": "pet_info"}
            },
            {
                "text": "Ted Sarandos and Greg Peters are now co-CEOs of Netflix, with Reed Hastings as Executive Chairman.",
                "category": "business",
                "priority": MemoryPriority.MEDIUM,
                "metadata": {"source": "news", "type": "company_update"}
            },
            {
                "text": "Using TensorFlow and PyTorch for deep learning projects requires understanding of neural networks.",
                "category": "technology",
                "priority": MemoryPriority.HIGH,
                "metadata": {"source": "technical", "type": "framework_info"}
            }
        ]
        
        conversation_id = "test_conversation_001"
        stored_ids = []
        
        for memory in entity_test_memories:
            memory_id = await memory_module.store_memory(
                text=memory["text"],
                metadata=memory["metadata"],
                priority=memory["priority"],
                conversation_id=conversation_id
            )
            stored_ids.append(memory_id)
            print(f"\nStored memory with ID: {memory_id}")
            print(f"Text: {memory['text']}")

        # Test Scenario 2: Memory Recall
        print("\n2. Testing Memory Recall...")
        test_queries = [
            "Tell me about Mimi the cat",
            "Who are Netflix's CEOs?",
            "What deep learning frameworks were mentioned?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            memories = await memory_module.recall_memories(
                query=query,
                conversation_id=conversation_id,
                top_k=2
            )
            
            if memories:
                print(f"Found {len(memories)} relevant memories:")
                for mem in memories:
                    print(f"- {mem['text']}")
                    print(f"  Relevance: {mem.get('relevance', 'N/A')}")
            else:
                print("No relevant memories found")

        # Test Scenario 3: Entity Relationships
        print("\n3. Testing Entity Relationships...")
        relationships = memory_module.get_entity_relationships()
        if relationships:
            print("\nFound Entity Relationships:")
            for rel in relationships[:3]:
                print(f"\nRelationship between '{rel['entity1']}' and '{rel['entity2']}':")
                print(f"Shared contexts: {len(rel['shared_contexts'])}")
                print(f"Sample context: {rel['shared_contexts'][0] if rel['shared_contexts'] else 'None'}")

        # Test Scenario 4: Memory Maintenance
        print("\n4. Testing Memory Maintenance...")
        # Trigger memory decay
        await memory_module._check_memory_decay()
        print("Memory decay check completed")
        
        # Test cleanup
        await memory_module.cleanup_memories()
        print("Memory cleanup completed")

        # Final Statistics
        print("\n5. Memory Statistics:")
        conversation_memories = await memory_module.recall_memories(
            query="",
            conversation_id=conversation_id,
            top_k=10
        )
        print(f"Total memories in conversation: {len(conversation_memories)}")
        
        # Cleanup test data
        print("\n6. Cleaning up test data...")
        for memory_id in stored_ids:
            # Add cleanup logic here if needed
            pass
        print("Cleanup completed")

        print("\nEpisodic Memory testing completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Ensure proper cleanup of resources
        if 'memory_module' in locals():
            await memory_module._cleanup()

if __name__ == "__main__":
    asyncio.run(test_memory_functionality()) 