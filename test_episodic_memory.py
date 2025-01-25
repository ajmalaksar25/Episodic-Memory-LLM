import asyncio
from datetime import datetime, timedelta
from example_usage import MemoryPriority
from episodic_memory import EpisodicMemoryLLM

async def test_ement_functionality():
    try:
        print("\n=== Testing EMENT Enhanced Functionality ===")
        
        # Initialize
        print("Initializing EpisodicMemoryLLM...")
        memory_llm = EpisodicMemoryLLM(collection_name="test_ement")
        
        # Clear existing data
        try:
            existing_docs = memory_llm.collection.get()
            if existing_docs['ids']:
                memory_llm.collection.delete(ids=existing_docs['ids'])
        except Exception as e:
            print(f"Cleanup initialization error (safe to ignore): {e}")
        
        # Test Scenario 1: Entity-Rich Content
        print("\n1. Testing Entity-Rich Content Storage...")
        entity_test_memories = [
            (
                "The user's cat Mimi is sick because she keeps eating lemons. "
                "Mimi is a ginger cat who loves swimming.",
                "pets",
                MemoryPriority.HIGH
            ),
            (
                "Ted Sarandos and Greg Peters are now co-CEOs of Netflix, "
                "with Reed Hastings as Executive Chairman.",
                "business",
                MemoryPriority.MEDIUM
            ),
            (
                "Using TensorFlow and PyTorch for deep learning projects "
                "requires understanding of neural networks.",
                "ai",
                MemoryPriority.HIGH
            )
        ]
        
        stored_ids = []
        for text, category, priority in entity_test_memories:
            memory_id = await memory_llm.store_memory(
                text,
                {"category": category},
                priority
            )
            stored_ids.append(memory_id)
            print(f"\nStored memory with ID: {memory_id}")
            print(f"Text: {text}")

        # Test Scenario 2: Entity Extraction
        print("\n2. Testing Entity Extraction...")
        entities = ["Mimi", "Netflix", "TensorFlow"]
        for entity in entities:
            summary = await memory_llm.summarize_entity_memories(entity)
            print(f"\nEntity: {entity}")
            print(f"Summary: {summary}")
            
            details = await memory_llm.get_entity_details(entity)
            print(f"Details: {details}")

        # Test Scenario 3: Embedding-Based Queries
        print("\n3. Testing Embedding-Based Queries...")
        test_queries = [
            ("How is the user's cat doing?", "pets"),
            ("What is Netflix's leadership structure?", "business"),
            ("Which deep learning frameworks are mentioned?", "ai")
        ]
        
        for query, category in test_queries:
            results = await memory_llm.recall_memories(
                query=query,
                top_k=2,
                where={"category": category}
            )
            print(f"\nQuery: {query}")
            print(f"Results ({len(results)} found):")
            for result in results:
                print(f"- {result['text']}")

        # Test Scenario 4: Combined EMENT Approach
        print("\n4. Testing Combined EMENT Approach...")
        test_cases = [
            {
                "query": "Tell me about Mimi's swimming habits",
                "expected_entities": ["Mimi", "swimming"],
                "category": "pets"
            },
            {
                "query": "Who are the CEOs of Netflix?",
                "expected_entities": ["Netflix", "Ted Sarandos", "Greg Peters"],
                "category": "business"
            },
            {
                "query": "What ML frameworks work with neural networks?",
                "expected_entities": ["TensorFlow", "PyTorch"],
                "category": "ai"
            }
        ]
        
        for case in test_cases:
            print(f"\nTest case: {case['query']}")
            
            # Get entity-based results
            entity_results = []
            for entity in case['expected_entities']:
                if entity in memory_llm.entity_dict:
                    entity_results.extend(memory_llm.entity_dict[entity])
            
            # Get embedding-based results
            embedding_results = await memory_llm.recall_memories(
                query=case['query'],
                top_k=3,
                where={"category": case['category']}
            )
            
            print("Entity-based results:")
            for result in entity_results:
                print(f"- {result}")
                
            print("\nEmbedding-based results:")
            for result in embedding_results:
                print(f"- {result['text']}")

        # Test Scenario 5: Memory Statistics
        print("\n5. Testing Memory Statistics...")
        stats = await memory_llm.get_enhanced_stats()
        print("\nMemory Statistics:")
        print(f"Total memories: {stats['total_memories']}")
        print(f"Categories: {stats['categories']}")
        print("\nEntity Analysis:")
        print(f"Total entities: {len(stats['entity_analysis']['entities_by_frequency'])}")
        print("\nTop entities by frequency:")
        for entity, count in list(stats['entity_analysis']['entities_by_frequency'].items())[:10]:
            print(f"- {entity}: {count} occurrences")

        # Cleanup
        print("\n6. Cleaning up...")
        try:
            all_docs = memory_llm.collection.get()
            if all_docs['ids']:
                memory_llm.collection.delete(ids=all_docs['ids'])
            print("Cleanup successful!")
        except Exception as e:
            print(f"Cleanup error: {e}")

        print("\nEMENT testing completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_ement_functionality()) 