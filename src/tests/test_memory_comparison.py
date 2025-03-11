import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_providers.groq_provider import GroqProvider
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from chatbot import SmartChatBot

# Load environment variables
load_dotenv()

class MemoryComparisonTest:
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768"):
        # Initialize the same LLM provider for both bots
        self.llm_provider = GroqProvider(api_key=api_key, model_name=model_name)
        
        # Initialize memory module for enhanced bot
        self.memory_module = EpisodicMemoryModule(
            llm_provider=self.llm_provider,
            collection_name="memory_test",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Initialize both chatbots
        self.memory_bot = SmartChatBot(
            llm_provider=self.llm_provider,
            memory_module=self.memory_module,
            persona="helpful AI assistant with memory"
        )
        
        # Standard bot without memory
        self.standard_bot = GroqProvider(api_key=api_key, model_name=model_name)
        
        self.results = {
            "standard": {"responses": [], "metrics": {}},
            "memory_enhanced": {"responses": [], "metrics": {}}
        }

    async def run_conversation_test(self, conversation: List[Dict]) -> Tuple[List[str], List[str]]:
        """Run the same conversation through both bots"""
        standard_responses = []
        memory_responses = []
        
        print("\nStarting conversation test...")
        
        # Start new conversation for memory bot
        await self.memory_bot.start_conversation()
        
        for message in conversation:
            print(f"\nUser: {message['text']}")
            
            # Get responses from both bots
            standard_response = await self.standard_bot.generate_response(message['text'])
            memory_response = await self.memory_bot.chat(message['text'])
            
            print(f"\nStandard Bot: {standard_response}")
            print(f"Memory Bot: {memory_response}")
            
            standard_responses.append(standard_response)
            memory_responses.append(memory_response)
            
            # Add delay between messages to simulate real conversation
            await asyncio.sleep(1)
        
        return standard_responses, memory_responses

    async def evaluate_context_retention(self, responses: List[str], context_questions: List[Dict]) -> Dict:
        """
        Evaluate how well the responses maintain context using EMENT-inspired metrics
        """
        try:
            metrics = {
                "context_accuracy": 0.0,
                "consistency_score": 0.0,
                "relevant_details_count": 0,
                "entity_retention": 0.0,
                "semantic_similarity": 0.0
            }
            
            if not responses or not context_questions:
                return metrics
            
            # 1. Context Accuracy Evaluation
            total_questions = len(context_questions)
            correct_contexts = 0
            
            for i, question in enumerate(context_questions):
                if i >= len(responses):
                    break
                
                # Check if response contains key information from previous context
                previous_context = " ".join([q["text"] for q in context_questions[:i]])
                response = responses[i]
                
                # Use LLM to evaluate context accuracy
                evaluation_prompt = f"""
                Previous Context: {previous_context}
                Question: {question['text']}
                Response: {response}
                
                Rate the response's accuracy in maintaining context from 0 to 1, where:
                0 = Completely ignores previous context
                1 = Perfectly maintains and uses previous context
                
                Return only the numerical score.
                """
                
                score = float(await self.llm_provider.generate_response(evaluation_prompt))
                correct_contexts += score
            
            metrics["context_accuracy"] = correct_contexts / total_questions if total_questions > 0 else 0
            
            # 2. Consistency Score
            inconsistencies = 0
            for i in range(1, len(responses)):
                consistency_prompt = f"""
                Response 1: {responses[i-1]}
                Response 2: {responses[i]}
                
                Rate the consistency between these responses from 0 to 1, where:
                0 = Completely contradictory
                1 = Perfectly consistent
                
                Consider:
                - Factual consistency
                - Logical flow
                - Entity references
                
                Return only the numerical score.
                """
                
                consistency = float(await self.llm_provider.generate_response(consistency_prompt))
                inconsistencies += consistency
            
            metrics["consistency_score"] = inconsistencies / (len(responses) - 1) if len(responses) > 1 else 1.0
            
            # 3. Relevant Details Count
            for response in responses:
                # Extract entities and key details
                analysis_prompt = f"""
                Analyze this response and count the number of specific, relevant details from previous context:
                {response}
                
                Return only the numerical count.
                """
                
                details_count = int(await self.llm_provider.generate_response(analysis_prompt))
                metrics["relevant_details_count"] += details_count
            
            # 4. Entity Retention (inspired by EMENT's entity tracking)
            total_entities = 0
            retained_entities = 0
            
            for i, question in enumerate(context_questions):
                if i >= len(responses):
                    break
                
                # Extract entities from question and previous context
                previous_entities = set()
                for prev_q in context_questions[:i+1]:
                    entities = await self._extract_entities(prev_q["text"])
                    previous_entities.update(entities)
                
                # Extract entities from response
                response_entities = await self._extract_entities(responses[i])
                
                if previous_entities:
                    total_entities += len(previous_entities)
                    retained_entities += len(previous_entities.intersection(response_entities))
            
            metrics["entity_retention"] = retained_entities / total_entities if total_entities > 0 else 0
            
            # 5. Semantic Similarity (using embeddings like EMENT)
            total_similarity = 0
            for i, question in enumerate(context_questions):
                if i >= len(responses):
                    break
                
                # Get embeddings for question and response
                question_embedding = await self._get_embedding(question["text"])
                response_embedding = await self._get_embedding(responses[i])
                
                # Calculate cosine similarity
                similarity = self._calculate_similarity(question_embedding, response_embedding)
                total_similarity += similarity
            
            metrics["semantic_similarity"] = total_similarity / len(responses) if responses else 0
            
            return metrics
        
        except Exception as e:
            print(f"Error in evaluate_context_retention: {e}")
            return {
                "context_accuracy": 0,
                "consistency_score": 0,
                "relevant_details_count": 0,
                "entity_retention": 0,
                "semantic_similarity": 0
            }

    async def _extract_entities(self, text: str) -> set:
        """Extract entities from text using spaCy"""
        try:
            doc = self.nlp(text)
            return {ent.text.lower() for ent in doc.ents}
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return set()

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        try:
            # Use the same embedding model as memory module
            return self.memory_module.get_embedding(text)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2:
                return 0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0

    def save_results(self, filename: str = "memory_comparison_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

async def main():
    try:
        # Initialize test suite
        tester = MemoryComparisonTest(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        
        # Test scenarios that evaluate context retention
        test_conversations = [
            {
                "name": "Technical Discussion",
                "messages": [
                    {"text": "I'm working on a machine learning project using PyTorch."},
                    {"text": "The main challenge is handling large datasets efficiently."},
                    {"text": "What preprocessing techniques would you recommend?"},
                    {"text": "Going back to the dataset issue we discussed earlier, any specific solutions?"}
                ]
            },
            {
                "name": "Complex Problem Solving",
                "messages": [
                    {"text": "We need to optimize our recommendation system."},
                    {"text": "Currently using collaborative filtering approach."},
                    {"text": "What are the pros and cons of our current approach?"},
                    {"text": "How would deep learning compare to what we discussed?"}
                ]
            }
        ]
        
        for scenario in test_conversations:
            print(f"\nTesting scenario: {scenario['name']}")
            standard_responses, memory_responses = await tester.run_conversation_test(scenario['messages'])
            
            # Evaluate responses
            standard_metrics = await tester.evaluate_context_retention(
                standard_responses,
                scenario['messages']
            )
            memory_metrics = await tester.evaluate_context_retention(
                memory_responses,
                scenario['messages']
            )
            
            # Store results
            tester.results["standard"]["metrics"][scenario["name"]] = standard_metrics
            tester.results["memory_enhanced"]["metrics"][scenario["name"]] = memory_metrics
            
            print(f"\nResults for {scenario['name']}:")
            print("Standard Bot Metrics:", standard_metrics)
            print("Memory Bot Metrics:", memory_metrics)
        
        # Save results
        tester.save_results()
        print("\nTest results saved to memory_comparison_results.json")
        
    except Exception as e:
        print(f"Error in memory comparison test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 