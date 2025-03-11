import asyncio
import json
import time
from datetime import datetime
import os
from typing import Dict, List
from dotenv import load_dotenv
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_providers.groq_provider import GroqProvider
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from chatbot import SmartChatBot

# Load environment variables
load_dotenv()

class ChatbotEvaluator:
    def __init__(self, api_key: str, model_name: str = "mixtral-8x7b-32768"):
        # Initialize providers
        self.baseline_llm = GroqProvider(api_key=api_key, model_name=model_name)
        self.memory_llm = GroqProvider(api_key=api_key, model_name=model_name)
        
        # Initialize memory module with proper entity config
        self.memory_module = EpisodicMemoryModule(
            llm_provider=self.memory_llm,
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
        
        # Initialize chatbots
        self.smart_chatbot = SmartChatBot(
            llm_provider=self.memory_llm,
            memory_module=self.memory_module,
            persona="helpful AI assistant"
        )
        
        self.results = {
            "baseline": {"responses": [], "metrics": {}},
            "memory_enhanced": {"responses": [], "metrics": {}}
        }

    async def run_memory_test_scenario(self, scenario: Dict):
        """Run a test scenario that evaluates memory retention"""
        print(f"\nRunning memory test scenario: {scenario['name']}")
        
        # First phase: Store information
        print("\nPhase 1: Storing information...")
        for message in scenario['context_messages']:
            await self.smart_chatbot.chat(message)
            
        # Wait a bit to simulate time passing
        await asyncio.sleep(1)
        
        # Second phase: Test recall
        print("\nPhase 2: Testing recall...")
        baseline_responses = []
        memory_responses = []
        
        for query in scenario['test_queries']:
            # Test baseline LLM
            baseline_start = time.time()
            baseline_response = await self.baseline_llm.generate_response(query)
            baseline_time = time.time() - baseline_start
            
            baseline_responses.append({
                "query": query,
                "response": baseline_response,
                "response_time": baseline_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Test memory-enhanced chatbot
            memory_start = time.time()
            memory_response = await self.smart_chatbot.chat(query)
            memory_time = time.time() - memory_start
            
            memory_responses.append({
                "query": query,
                "response": memory_response,
                "response_time": memory_time,
                "timestamp": datetime.now().isoformat()
            })
        
        # Store results
        self.results["baseline"]["responses"].extend(baseline_responses)
        self.results["memory_enhanced"]["responses"].extend(memory_responses)
        
        # Calculate metrics
        self.results["baseline"]["metrics"][scenario["name"]] = await self._calculate_metrics(baseline_responses)
        self.results["memory_enhanced"]["metrics"][scenario["name"]] = await self._calculate_metrics(memory_responses)
        
        return baseline_responses, memory_responses

    async def _calculate_metrics(self, responses: List[Dict]) -> Dict:
        """Calculate comprehensive metrics for responses"""
        total_time = sum(r["response_time"] for r in responses)
        avg_length = sum(len(r["response"]) for r in responses) / len(responses)
        
        # Calculate information retention score
        retention_score = await self._evaluate_information_retention(responses)
        
        # Calculate consistency score
        consistency_score = await self.evaluate_consistency(responses)
        
        return {
            "response_time": total_time,
            "avg_response_length": avg_length,
            "retention_score": retention_score,
            "consistency_score": consistency_score
        }

    async def _evaluate_information_retention(self, responses: List[Dict]) -> float:
        """Evaluate how well information from context is retained in responses"""
        try:
            prompt = f"""
            Evaluate how well these responses retain and use contextual information.
            Score from 0-1 where 1 means perfect information retention and relevant usage.
            
            Responses to evaluate:
            {json.dumps([{
                'query': r['query'],
                'response': r['response']
            } for r in responses], indent=2)}
            
            Return only a number between 0 and 1.
            """
            
            score_str = await self.baseline_llm.generate_response(prompt)
            try:
                return float(score_str.strip())
            except ValueError:
                print(f"Error parsing retention score: {score_str}")
                return 0
                
        except Exception as e:
            print(f"Error evaluating retention: {e}")
            return 0

    async def evaluate_consistency(self, responses: List[Dict]) -> float:
        """Evaluate response consistency"""
        try:
            consistency_scores = []
            for i in range(len(responses) - 1):
                prompt = f"""
                Rate the consistency between these responses (0-1):
                Query 1: {responses[i]['query']}
                Response 1: {responses[i]['response']}
                
                Query 2: {responses[i + 1]['query']}
                Response 2: {responses[i + 1]['response']}
                
                Return only a number between 0 and 1.
                """
                
                score_str = await self.baseline_llm.generate_response(prompt)
                try:
                    score = float(score_str.strip())
                    consistency_scores.append(score)
                except ValueError:
                    print(f"Error parsing consistency score: {score_str}")
                    
            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            
        except Exception as e:
            print(f"Error evaluating consistency: {e}")
            return 0

    def save_results(self, filename: str = "chatbot_evaluation_results.json"):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

async def main():
    # Define memory test scenarios
    memory_test_scenarios = [
        {
            "name": "Technical Concept Memory",
            "context_messages": [
                "Let me tell you about a new machine learning technique called 'gradient boosting'.",
                "Gradient boosting combines weak learners sequentially to create a strong model.",
                "It's particularly effective for structured data and uses decision trees as base learners."
            ],
            "test_queries": [
                "What was the machine learning technique we discussed earlier?",
                "How does it work?",
                "What are its main applications?"
            ]
        },
        {
            "name": "Project Context Memory",
            "context_messages": [
                "I'm working on a Python project using FastAPI and PostgreSQL.",
                "The main challenge is handling real-time updates with WebSocket connections.",
                "We're using Redis for caching to improve performance."
            ],
            "test_queries": [
                "What database am I using in my project?",
                "How am I handling real-time updates?",
                "What's the caching solution we discussed?"
            ]
        },
        {
            "name": "Complex Problem Memory",
            "context_messages": [
                "We need to optimize our recommendation system that handles millions of users.",
                "Current approach uses collaborative filtering with matrix factorization.",
                "We're considering switching to deep learning with embedding layers."
            ],
            "test_queries": [
                "What was the scale of our recommendation system?",
                "What approach are we currently using?",
                "What alternative solution was proposed?"
            ]
        }
    ]
    
    try:
        # Initialize evaluator
        evaluator = ChatbotEvaluator(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        
        # Run memory test scenarios
        for scenario in memory_test_scenarios:
            baseline_responses, memory_responses = await evaluator.run_memory_test_scenario(scenario)
            
            print(f"\nResults for {scenario['name']}:")
            print("\nBaseline Metrics:")
            print(json.dumps(evaluator.results["baseline"]["metrics"][scenario["name"]], indent=2))
            print("\nMemory-Enhanced Metrics:")
            print(json.dumps(evaluator.results["memory_enhanced"]["metrics"][scenario["name"]], indent=2))
        
        # Save results
        evaluator.save_results()
        print("\nEvaluation complete. Results saved to chatbot_evaluation_results.json")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 