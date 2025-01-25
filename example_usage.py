import asyncio
import os
import json
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from llm_providers import GroqProvider
async def main():
    # Ensure GROQ_API_KEY is set
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("MODEL_NAME") or "mixtral-8x7b-32768"
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    try:
        # Initialize Groq provider
        groq_provider = GroqProvider(
            api_key=api_key,
            model_name=model_name
        )
        
        # Initialize memory module
        memory_module = EpisodicMemoryModule(
            llm_provider=groq_provider,
            decay_interval_hours=24,
            min_references_to_keep=2
        )
        
        # Example conversation
        conversation_id = "conv_001"
        
        # Test connection
        test_response = await groq_provider.generate_response("Test connection")
        print("Connection test:", "Success" if test_response else "Failed")
        
        # Store test memories
        memories_to_store = [
            {
                "text": "User asked about implementing neural networks in PyTorch",
                "metadata": {
                    "category": "programming",
                    "subcategory": "deep_learning"
                },
                "priority": MemoryPriority.HIGH
            },
            {
                "text": "Explained backpropagation and gradient descent concepts",
                "metadata": {
                    "category": "machine_learning",
                    "subcategory": "neural_networks"
                },
                "priority": MemoryPriority.HIGH
            }
        ]

        print("\nStoring test memories...")
        stored_ids = []
        for memory in memories_to_store:
            try:
                memory_id = await memory_module.store_memory(
                    text=memory["text"],
                    metadata=memory["metadata"],
                    priority=memory["priority"],
                    conversation_id="conv_001"
                )
                if memory_id:
                    stored_ids.append(memory_id)
                    print(f"Successfully stored memory: {memory_id}")
            except Exception as e:
                print(f"Error storing memory: {e}")

        print("\nCleaning up duplicates...")
        await memory_module.cleanup_memories()

        print("\nTesting memory recall...")
        # Test different recall scenarios
        queries = [
            ("neural network implementation", "programming"),
            ("machine learning concepts", "machine_learning"),
            ("PyTorch usage", None)
        ]

        for query, category in queries:
            print(f"\nRecalling memories for query: '{query}' (category: {category})")
            memories = await memory_module.recall_memories(
                query=query,
                top_k=3,
                category=category,
                conversation_id="conv_001",
                include_related=True
            )
            print(f"Found {len(memories)} memories:")
            for mem in memories:
                print(f"- {mem['text']} (importance: {mem.get('importance', 'N/A')})")

        print("\nGetting memory statistics...")
        stats = memory_module.get_memory_stats()
        print("Memory Statistics:", json.dumps(stats, indent=2))

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 