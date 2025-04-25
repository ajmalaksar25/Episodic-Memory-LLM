import asyncio
import time
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from chatbot import SmartChatBot

class BenchmarkMetrics:
    def __init__(self, name: str):
        self.name = name
        self.response_times = []
        self.memory_accuracies = []
        self.context_scores = []
        self.token_counts = []
        self.relevance_scores = []
        self.entity_recall_scores = []
        self.temporal_consistency_scores = []
        self.factual_accuracy_scores = []
        self.conversation_coherence_scores = []
        
        # New metrics for measuring factual question handling
        self.factual_question_accuracy_scores = []  # Accuracy on factual questions
        self.factual_recall_scores = []  # Ability to recall specific facts
        self.consistency_over_time_scores = []  # Consistency in answers across conversations
        self.factual_precision_scores = []  # Precision of factual answers
        
    def add_metric(self, metric_type: str, value: float):
        if hasattr(self, metric_type):
            getattr(self, metric_type).append(value)

    def get_average(self, metric_type: str) -> float:
        if hasattr(self, metric_type):
            values = getattr(self, metric_type)
            return sum(values) / len(values) if values else 0
        return 0
    
    def get_improvement_percentage(self, other_metrics, metric_type: str) -> float:
        """Calculate percentage improvement over another metrics object"""
        if hasattr(self, metric_type) and hasattr(other_metrics, metric_type):
            self_avg = self.get_average(metric_type)
            other_avg = other_metrics.get_average(metric_type)
            
            if other_avg == 0:
                return 0
            
            return ((self_avg - other_avg) / other_avg) * 100
        return 0

    def get_metrics_dict(self) -> Dict[str, Any]:
        return {
            'response_times': self.response_times,
            'response_time_avg': self.get_average('response_times'),
            'memory_accuracies': self.memory_accuracies,
            'memory_accuracy_avg': self.get_average('memory_accuracies'),
            'context_scores': self.context_scores,
            'context_score_avg': self.get_average('context_scores'),
            'token_counts': self.token_counts,
            'token_count_avg': self.get_average('token_counts'),
            'relevance_scores': self.relevance_scores,
            'relevance_score_avg': self.get_average('relevance_scores'),
            'entity_recall_scores': self.entity_recall_scores,
            'entity_recall_avg': self.get_average('entity_recall_scores'),
            'temporal_consistency_scores': self.temporal_consistency_scores,
            'temporal_consistency_avg': self.get_average('temporal_consistency_scores'),
            'factual_accuracy_scores': self.factual_accuracy_scores,
            'factual_accuracy_avg': self.get_average('factual_accuracy_scores'),
            'conversation_coherence_scores': self.conversation_coherence_scores,
            'conversation_coherence_avg': self.get_average('conversation_coherence_scores')
        }
        
    def get_metric(self, metric_name: str, default_value: Any = None) -> Any:
        """Get a specific metric value by name, with fallback to default value"""
        metrics_dict = self.get_metrics_dict()
        return metrics_dict.get(metric_name, default_value)
        
    def get_metric_list(self, metric_name: str, default_value: List = None) -> List:
        """Get a list-type metric by name, with fallback to default value"""
        if default_value is None:
            default_value = []
            
        metrics_dict = self.get_metrics_dict()
        return metrics_dict.get(metric_name, default_value)

class BenchmarkingSystem:
    def __init__(self, llm_provider, test_scenarios: List[Dict[str, Any]], model_name: str = "llama3-8b-8192"):
        """Initialize the benchmarking system"""
        self.llm_provider = llm_provider
        self.test_scenarios = test_scenarios
        self.model_name = model_name
        
        # Initialize metrics
        self.baseline_metrics = BenchmarkMetrics(name=f"{model_name}_baseline")
        self.episodic_metrics = BenchmarkMetrics(name=f"{model_name}_episodic")
        
        # Store responses for analysis
        self.baseline_responses = []
        self.episodic_responses = []
        
        # Conversation history for context
        self.conversation_history = [[] for _ in range(len(test_scenarios))]
        
        # Cache for performance
        self._cache = {}
        
    async def run_traditional_benchmark(self):
        """Benchmark traditional LLM with conversation history"""
        try:
            # Create a dummy memory module for traditional benchmark
            # This avoids NoneType errors in SmartChatBot
            dummy_memory_module = EpisodicMemoryModule(
                llm_provider=self.llm_provider,
                collection_name=f"traditional_benchmark_{self.model_name}",
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Initialize chatbot with dummy memory module
            chatbot = SmartChatBot(
                llm_provider=self.llm_provider,
                memory_module=dummy_memory_module,
                persona="helpful AI assistant"
            )
            
            # Track conversation history for each scenario
            for scenario_idx, scenario in enumerate(self.test_scenarios):
                print(f"Running traditional benchmark for scenario {scenario_idx+1}/{len(self.test_scenarios)}")
                
                # Set the conversation ID for this scenario
                conversation_id = f"traditional_benchmark_scenario_{scenario_idx}"
                await chatbot.switch_conversation(conversation_id)
                
                # Initialize conversation for this scenario if needed
                if scenario_idx not in self.conversation_history:
                    self.conversation_history[scenario_idx] = []
                
                # Add context to conversation history if it's a follow-up question
                if scenario.get('is_followup', False) and self.conversation_history[scenario_idx]:
                    for message in self.conversation_history[scenario_idx]:
                        if message['role'] == 'user':
                            try:
                                await chatbot.chat(message['content'])
                            except Exception as e:
                                print(f"Error in chat processing: {e}")
                
                # Process the current query
                start_time = time.time()
                try:
                    response = await chatbot.chat(scenario['input'])
                except Exception as e:
                    print(f"Error in chat processing: {e}")
                    response = "Error processing this query."
                end_time = time.time()
                
                # Record conversation
                self.conversation_history[scenario_idx].append({'role': 'user', 'content': scenario['input']})
                self.conversation_history[scenario_idx].append({'role': 'assistant', 'content': response})
                
                # Record metrics
                self.baseline_metrics.add_metric('response_times', end_time - start_time)
                self.baseline_metrics.add_metric('relevance_scores', 
                    self._calculate_relevance(response, scenario['expected_output']))
                self.baseline_metrics.add_metric('context_scores',
                    self._calculate_context_preservation(response, scenario['context']))
                
                # Calculate entity recall
                if 'key_entities' in scenario:
                    self.baseline_metrics.add_metric('entity_recall_scores',
                        self._calculate_entity_recall(response, scenario.get('key_entities', [])))
                
                # Calculate temporal consistency for follow-up questions
                if scenario.get('is_followup', False) and 'previous_context' in scenario:
                    self.baseline_metrics.add_metric('temporal_consistency_scores',
                        self._calculate_temporal_consistency(response, scenario.get('previous_context', '')))
                
                # Calculate factual accuracy
                self.baseline_metrics.add_metric('factual_accuracy_scores',
                    self._calculate_factual_accuracy(response, scenario['expected_output']))
                
                # Add new factual metrics if the scenario has the necessary data
                if 'question_type' in scenario:
                    self.baseline_metrics.add_metric('factual_question_accuracy_scores',
                        self._calculate_factual_question_accuracy(
                            response, 
                            scenario['expected_output'],
                            scenario.get('question_type', 'general')
                        )
                    )
                
                if 'key_facts' in scenario:
                    self.baseline_metrics.add_metric('factual_recall_scores',
                        self._calculate_factual_recall(
                            response, 
                            scenario.get('key_facts', []),
                            scenario.get('previously_mentioned', False)
                        )
                    )
                
                # Calculate consistency over time if we have previous responses
                if scenario.get('is_followup', False) and len(self.baseline_responses) > 0:
                    previous_responses = [r for r in self.baseline_responses if r]
                    if previous_responses and 'key_facts' in scenario:
                        self.baseline_metrics.add_metric('consistency_over_time_scores',
                            self._calculate_consistency_over_time(
                                response,
                                previous_responses,
                                scenario.get('key_facts', [])
                            )
                        )
                
                # Store the response for later comparisons
                self.baseline_responses.append(response)
                
                # Calculate conversation coherence
                if len(self.conversation_history[scenario_idx]) >= 4:  # At least 2 exchanges
                    self.baseline_metrics.add_metric('conversation_coherence_scores',
                        self._calculate_conversation_coherence(self.conversation_history[scenario_idx]))
        except Exception as e:
            print(f"Error in traditional benchmark: {e}")
            import traceback
            traceback.print_exc()
            
    async def run_episodic_benchmark(self, memory_module: EpisodicMemoryModule):
        """Benchmark LLM with Episodic Memory Module"""
        try:
            # First, seed the memory module with context from scenarios
            await self._seed_memory_module(memory_module)
            
            # Initialize chatbot with memory module
            chatbot = SmartChatBot(
                llm_provider=self.llm_provider,
                memory_module=memory_module,
                persona="helpful AI assistant"
            )
            
            # Track conversation history for each scenario
            for scenario_idx, scenario in enumerate(self.test_scenarios):
                print(f"Running episodic benchmark for scenario {scenario_idx+1}/{len(self.test_scenarios)}")
                
                # Set the conversation ID for this scenario
                conversation_id = f"benchmark_scenario_{scenario_idx}"
                await chatbot.switch_conversation(conversation_id)
                
                # Initialize conversation for this scenario if needed
                if scenario_idx not in self.conversation_history:
                    self.conversation_history[scenario_idx] = []
                
                # Add context to conversation history if it's a follow-up question
                if scenario.get('is_followup', False) and self.conversation_history[scenario_idx]:
                    for message in self.conversation_history[scenario_idx]:
                        if message['role'] == 'user':
                            try:
                                await chatbot.chat(message['content'])
                            except Exception as e:
                                print(f"Error in chat processing: {e}")
                
                # Process the current query
                start_time = time.time()
                try:
                    response = await chatbot.chat(scenario['input'])
                except Exception as e:
                    print(f"Error in chat processing: {e}")
                    response = "Error processing this query."
                end_time = time.time()
                
                # Record conversation
                self.conversation_history[scenario_idx].append({'role': 'user', 'content': scenario['input']})
                self.conversation_history[scenario_idx].append({'role': 'assistant', 'content': response})
                
                # Record metrics
                self.episodic_metrics.add_metric('response_times', end_time - start_time)
                self.episodic_metrics.add_metric('relevance_scores',
                    self._calculate_relevance(response, scenario['expected_output']))
                self.episodic_metrics.add_metric('context_scores',
                    self._calculate_context_preservation(response, scenario['context']))
                
                # Calculate entity recall
                if 'key_entities' in scenario:
                    self.episodic_metrics.add_metric('entity_recall_scores',
                        self._calculate_entity_recall(response, scenario.get('key_entities', [])))
                
                # Calculate temporal consistency for follow-up questions
                if scenario.get('is_followup', False) and 'previous_context' in scenario:
                    self.episodic_metrics.add_metric('temporal_consistency_scores',
                        self._calculate_temporal_consistency(response, scenario.get('previous_context', '')))
                
                # Calculate factual accuracy
                self.episodic_metrics.add_metric('factual_accuracy_scores',
                    self._calculate_factual_accuracy(response, scenario['expected_output']))
                
                # Add new factual metrics if the scenario has the necessary data
                if 'question_type' in scenario:
                    self.episodic_metrics.add_metric('factual_question_accuracy_scores',
                        self._calculate_factual_question_accuracy(
                            response, 
                            scenario['expected_output'],
                            scenario.get('question_type', 'general')
                        )
                    )
                
                if 'key_facts' in scenario:
                    self.episodic_metrics.add_metric('factual_recall_scores',
                        self._calculate_factual_recall(
                            response, 
                            scenario.get('key_facts', []),
                            scenario.get('previously_mentioned', False)
                        )
                    )
                
                # Calculate consistency over time if we have previous responses
                if scenario.get('is_followup', False) and len(self.episodic_responses) > 0:
                    previous_responses = [r for r in self.episodic_responses if r]
                    if previous_responses and 'key_facts' in scenario:
                        self.episodic_metrics.add_metric('consistency_over_time_scores',
                            self._calculate_consistency_over_time(
                                response,
                                previous_responses,
                                scenario.get('key_facts', [])
                            )
                        )
                
                # Store the response for later comparisons
                self.episodic_responses.append(response)
                
                # Calculate conversation coherence
                if len(self.conversation_history[scenario_idx]) >= 4:  # At least 2 exchanges
                    self.episodic_metrics.add_metric('conversation_coherence_scores',
                        self._calculate_conversation_coherence(self.conversation_history[scenario_idx]))
                
                # Get memory accuracy
                try:
                    # Use the correct conversation ID when recalling memories
                    memories = await memory_module.recall_memories(
                        query=scenario['input'],
                        conversation_id=conversation_id
                    )
                    self.episodic_metrics.add_metric('memory_accuracies',
                        self._calculate_memory_accuracy(memories, scenario['context']))
                except Exception as e:
                    print(f"Error recalling memories: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error in episodic benchmark: {e}")
            import traceback
            traceback.print_exc()

    async def _seed_memory_module(self, memory_module: EpisodicMemoryModule):
        """Seed the memory module with context from scenarios"""
        try:
            for scenario_idx, scenario in enumerate(self.test_scenarios):
                if 'seed_context' in scenario:
                    print(f"Seeding memory with context from scenario {scenario_idx+1}")
                    # Add seed context to memory
                    try:
                        # Create a conversation ID for this scenario if needed
                        conversation_id = f"benchmark_scenario_{scenario_idx}"
                        
                        # Store the memory with the correct parameters
                        await memory_module.store_memory(
                            text=scenario['seed_context'],
                            metadata={
                                'type': 'context',
                                'timestamp': datetime.now().isoformat(),
                            },
                            conversation_id=conversation_id,
                            priority=MemoryPriority.HIGH
                        )
                    except Exception as e:
                        print(f"Error storing memory for scenario {scenario_idx+1}: {e}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"Error seeding memory module: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_relevance(self, response: str, expected: str) -> float:
        """Calculate relevance score using word overlap"""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        if not expected_words:
            return 0.0
        return len(response_words.intersection(expected_words)) / len(expected_words)

    def _calculate_context_preservation(self, response: str, context: str) -> float:
        """
        Calculate context preservation score with improved semantic matching
        
        Args:
            response: The model's response text
            context: The context provided to the model
            
        Returns:
            A score between 0 and 1 indicating how well the context was preserved
        """
        if not context or not response:
            return 0.0
        
        # Normalize text
        context_lower = context.lower()
        response_lower = response.lower()
        
        # Extract key phrases (2-3 word sequences) from context
        context_words = context_lower.split()
        context_phrases = []
        
        # Single words (excluding common stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        context_keywords = [word for word in context_words if word not in stopwords and len(word) > 3]
        
        # 2-word phrases
        if len(context_words) >= 2:
            for i in range(len(context_words) - 1):
                phrase = f"{context_words[i]} {context_words[i+1]}"
                context_phrases.append(phrase)
        
        # 3-word phrases
        if len(context_words) >= 3:
            for i in range(len(context_words) - 2):
                phrase = f"{context_words[i]} {context_words[i+1]} {context_words[i+2]}"
                context_phrases.append(phrase)
        
        # Count matches
        keyword_matches = sum(1 for keyword in context_keywords if keyword in response_lower)
        phrase_matches = sum(1 for phrase in context_phrases if phrase in response_lower)
        
        # Calculate scores
        keyword_score = keyword_matches / max(1, len(context_keywords))
        phrase_score = phrase_matches / max(1, len(context_phrases))
        
        # Combine scores (phrases are more important than single keywords)
        combined_score = (keyword_score * 0.4) + (phrase_score * 0.6)
        
        return combined_score

    def _calculate_memory_accuracy(self, memories: List[Dict], context: str) -> float:
        """Calculate memory accuracy score"""
        if not memories or not context:
            return 0.0
        
        context_words = set(context.lower().split())
        
        # Extract text from memories, handling different possible structures
        memory_texts = []
        for memory in memories:
            if isinstance(memory, dict):
                # Try different possible keys for the text content
                if 'text' in memory:
                    memory_texts.append(memory['text'])
                elif 'content' in memory:
                    memory_texts.append(memory['content'])
                elif 'memory' in memory:
                    memory_texts.append(memory['memory'])
            elif isinstance(memory, str):
                memory_texts.append(memory)
        
        if not memory_texts:
            return 0.0
            
        memory_words = set(' '.join(memory_texts).lower().split())
        
        if not context_words:
            return 0.0
            
        return len(memory_words.intersection(context_words)) / len(context_words)
    
    def _calculate_entity_recall(self, response: str, key_entities: List[str]) -> float:
        """
        Calculate entity recall score with improved matching
        
        Args:
            response: The model's response text
            key_entities: List of key entities that should be recalled
            
        Returns:
            A score between 0 and 1 indicating how well entities were recalled
        """
        if not key_entities:
            return 0.0
        
        response_lower = response.lower()
        found_entities = 0
        
        for entity in key_entities:
            entity_lower = entity.lower()
            
            # Check for exact match
            if entity_lower in response_lower:
                found_entities += 1
                continue
            
            # Check for partial matches for longer entities (names, organizations, etc.)
            if len(entity_lower.split()) > 1:
                # For multi-word entities, check if most words are present
                entity_parts = entity_lower.split()
                matched_parts = sum(1 for part in entity_parts if part in response_lower and len(part) > 3)
                if matched_parts >= len(entity_parts) * 0.75:  # 75% of words match
                    found_entities += 0.75
                    continue
            
            # Check for alternative forms or references
            # For example, "John Smith" might be referred to as "John" or "Mr. Smith"
            if len(entity_lower.split()) > 1:
                for part in entity_lower.split():
                    if len(part) > 3 and part in response_lower:
                        found_entities += 0.5 / len(entity_lower.split())
        
        # Normalize to 0-1 range
        recall_score = min(1.0, found_entities / len(key_entities))
        
        return recall_score
    
    def _calculate_temporal_consistency(self, response: str, previous_context: str) -> float:
        """Calculate temporal consistency score"""
        if not previous_context:
            return 0.0
        
        previous_words = set(previous_context.lower().split())
        response_words = set(response.lower().split())
        return len(response_words.intersection(previous_words)) / len(previous_words)
    
    def _calculate_factual_accuracy(self, response: str, expected_output: str) -> float:
        """Calculate factual accuracy score"""
        # This is a simplified version - in a real system, you might use
        # more sophisticated NLP techniques or another LLM to evaluate
        response_lower = response.lower()
        expected_lower = expected_output.lower()
        
        # Check for key phrases or facts
        key_phrases = expected_lower.split('. ')
        if not key_phrases:
            return 0.0
        
        found_phrases = sum(1 for phrase in key_phrases if phrase and phrase in response_lower)
        return found_phrases / len(key_phrases)
    
    def _calculate_factual_question_accuracy(self, response: str, expected_answer: str, 
                                            question_type: str = "general") -> float:
        """
        Calculate accuracy for factual question answering
        
        Args:
            response: The model's response
            expected_answer: The expected correct answer
            question_type: Type of factual question (general, specific, numerical, etc.)
            
        Returns:
            Accuracy score between 0.0 and 1.0
        """
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Different scoring based on question type
        if question_type == "numerical":
            # For numerical questions, extract numbers and compare
            import re
            response_numbers = re.findall(r'\d+\.?\d*', response_lower)
            expected_numbers = re.findall(r'\d+\.?\d*', expected_lower)
            
            if not expected_numbers:
                return 0.0
                
            if not response_numbers:
                return 0.0
                
            # Check if any of the response numbers match any expected numbers
            for r_num in response_numbers:
                if r_num in expected_numbers:
                    return 1.0
            return 0.0
            
        elif question_type == "specific":
            # For specific factual questions, look for exact match of key terms
            key_terms = expected_lower.split(',')
            matches = sum(1 for term in key_terms if term.strip() and term.strip() in response_lower)
            return matches / len(key_terms) if key_terms else 0.0
            
        else:  # general factual questions
            # Use semantic similarity for general questions
            key_statements = expected_lower.split('. ')
            score = 0.0
            
            for statement in key_statements:
                if statement and statement in response_lower:
                    score += 1.0
                # Partial credit for similar statements
                elif statement and len(statement) > 10:
                    # Check for partial matches by looking at significant overlapping words
                    statement_words = set(statement.split())
                    if len(statement_words) > 0:
                        overlap = sum(1 for word in statement_words 
                                     if word and len(word) > 3 and word in response_lower)
                        if overlap / len(statement_words) > 0.5:  # More than 50% overlap
                            score += 0.5
            
            return score / len(key_statements) if key_statements else 0.0
    
    def _calculate_factual_recall(self, response: str, key_facts: List[str], 
                                 previously_mentioned: bool = False) -> float:
        """
        Calculate recall of specific facts that were previously mentioned
        
        Args:
            response: The model's response
            key_facts: List of key facts to look for
            previously_mentioned: Whether these facts were mentioned earlier in the conversation
            
        Returns:
            Recall score between 0.0 and 1.0
        """
        if not key_facts:
            return 0.0
            
        response_lower = response.lower()
        
        # Count how many key facts are recalled in the response
        recalled_facts = sum(1 for fact in key_facts 
                            if fact and fact.lower() in response_lower)
        
        # If facts were previously mentioned, we expect higher recall
        recall_score = recalled_facts / len(key_facts)
        
        # Apply a penalty if facts were previously mentioned but not recalled
        if previously_mentioned and recall_score < 0.5:
            recall_score *= 0.8  # 20% penalty for poor recall of previously mentioned facts
            
        return recall_score
        
    def _calculate_consistency_over_time(self, current_response: str, previous_responses: List[str], 
                                       key_facts: List[str]) -> float:
        """
        Calculate consistency of factual responses over time
        
        Args:
            current_response: The current model response
            previous_responses: List of previous responses on the same topic
            key_facts: List of key facts to check for consistency
            
        Returns:
            Consistency score between 0.0 and 1.0
        """
        if not previous_responses or not key_facts:
            return 1.0  # No previous responses to compare against
            
        current_lower = current_response.lower()
        
        consistency_scores = []
        
        for fact in key_facts:
            if not fact:
                continue
                
            fact_lower = fact.lower()
            fact_in_current = fact_lower in current_lower
            
            # Check if the fact was mentioned consistently in previous responses
            fact_mentions = [1 if fact_lower in resp.lower() else 0 
                          for resp in previous_responses]
            
            if sum(fact_mentions) == 0 and not fact_in_current:
                # Fact was never mentioned - consistent but not informative
                consistency_scores.append(0.5)
            elif sum(fact_mentions) > 0 and fact_in_current:
                # Fact was mentioned before and now - consistent
                consistency_scores.append(1.0)
            elif sum(fact_mentions) > len(previous_responses) / 2 and not fact_in_current:
                # Fact was mentioned in majority of previous responses but not now - inconsistent
                consistency_scores.append(0.0)
            else:
                # Fact was mentioned in some but not most previous responses
                consistency_scores.append(0.5)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_conversation_coherence(self, conversation: List[Dict]) -> float:
        """Calculate conversation coherence score"""
        if len(conversation) < 4:  # Need at least 2 exchanges
            return 0.5  # Default medium coherence
        
        # Check coherence between non-adjacent exchanges
        coherence_score = 0
        
        for i in range(0, len(conversation) - 3, 2):
            # Compare user message i with assistant response i+3
            # This checks if later responses are consistent with earlier context
            user_msg = conversation[i]['content']
            later_response = conversation[i+3]['content']
            
            # Calculate coherence as context preservation
            exchange_coherence = self._calculate_context_preservation(later_response, user_msg)
            coherence_score += exchange_coherence
        
        # Average coherence score
        num_checks = max(1, (len(conversation) - 3) // 2)
        return coherence_score / num_checks

    def generate_visualizations(self, output_dir: str) -> List[str]:
        """Generate and save visualization files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        try:
            # Generate accuracy comparison
            if hasattr(self, '_generate_accuracy_comparison'):
                accuracy_file = self._generate_accuracy_comparison(output_dir)
                if accuracy_file:
                    saved_files.append(accuracy_file)
            
            # Generate factual recall comparison
            if hasattr(self, '_generate_factual_recall_visualization'):
                factual_file = self._generate_factual_recall_visualization(output_dir)
                if factual_file:
                    saved_files.append(factual_file)
            
            # Generate memory usage visualization
            if hasattr(self, '_generate_memory_usage_visualization'):
                memory_file = self._generate_memory_usage_visualization(output_dir)
                if memory_file:
                    saved_files.append(memory_file)
            
            # Generate response quality visualization
            if hasattr(self, '_generate_response_quality_visualization'):
                quality_file = self._generate_response_quality_visualization(output_dir)
                if quality_file:
                    saved_files.append(quality_file)
                
            # Generate model performance radar chart
            if hasattr(self, '_generate_performance_radar_chart'):
                radar_file = self._generate_performance_radar_chart(output_dir)
                if radar_file:
                    saved_files.append(radar_file)
                
            # Generate conversations coherence chart
            if hasattr(self, '_generate_conversation_coherence_chart'):
                coherence_file = self._generate_conversation_coherence_chart(output_dir)
                if coherence_file:
                    saved_files.append(coherence_file)
                
            return saved_files
        
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return saved_files

    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get benchmark results as a structured dictionary"""
        results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "baseline_metrics": self.baseline_metrics.get_metrics_dict(),
            "episodic_metrics": self.episodic_metrics.get_metrics_dict(),
            "improvement_percentages": self._calculate_improvement_percentages(),
            "baseline_responses": self.baseline_responses,
            "episodic_responses": self.episodic_responses,
            "test_scenarios_count": len(self.test_scenarios)
        }
        return results
    
    def _calculate_improvement_percentages(self) -> Dict[str, float]:
        """Calculate improvement percentages between baseline and episodic metrics"""
        improvements = {}
        baseline_dict = self.baseline_metrics.get_metrics_dict()
        episodic_dict = self.episodic_metrics.get_metrics_dict()
        
        for metric in baseline_dict:
            if isinstance(baseline_dict[metric], (int, float)) and isinstance(episodic_dict.get(metric), (int, float)):
                baseline_val = baseline_dict[metric]
                episodic_val = episodic_dict.get(metric, 0)
                
                if baseline_val != 0:
                    improvement = ((episodic_val - baseline_val) / baseline_val) * 100
                else:
                    improvement = 0 if episodic_val == 0 else 100  # If baseline is 0, use 0% or 100% improvement
                improvements[metric] = round(improvement, 2)
        
        return improvements

    @staticmethod
    def generate_multi_model_comparison(output_dir: str, model_results: Dict[str, Dict], metric_names: List[str] = None) -> str:
        """
        Generate a visualization comparing multiple models on specified metrics.
        
        Args:
            output_dir: Directory to save the visualization
            model_results: Dictionary mapping model names to their benchmark results
            metric_names: List of metric names to compare (defaults to key metrics if None)
            
        Returns:
            Path to the saved visualization file
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Default metrics if none provided
        if not metric_names:
            metric_names = ["accuracy", "context_relevance", "coherence", "memory_usage"]
        
        # Extract model names and data
        model_names = list(model_results.keys())
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2, figure=fig)
        
        # Prepare data for radar chart (overall performance)
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        
        # Set number of metrics
        num_metrics = len(metric_names)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot radar chart for each model
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, metric_names)
        
        for model_name in model_names:
            baseline = model_results[model_name].get("baseline_metrics", {})
            episodic = model_results[model_name].get("episodic_metrics", {})
            
            # Get values for this model
            baseline_values = [baseline.get(metric, 0) for metric in metric_names]
            episodic_values = [episodic.get(metric, 0) for metric in metric_names]
            
            # Close the loop
            baseline_values += baseline_values[:1]
            episodic_values += episodic_values[:1]
            
            # Plot baseline and episodic values
            ax_radar.plot(angles, baseline_values, label=f"{model_name} (Baseline)")
            ax_radar.plot(angles, episodic_values, label=f"{model_name} (Episodic)", linestyle='--')
            ax_radar.fill(angles, episodic_values, alpha=0.1)
        
        ax_radar.set_title("Model Performance Comparison (Radar)", fontsize=14)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Bar chart for improvement percentages
        ax_improvement = fig.add_subplot(gs[0, 1])
        
        # Calculate improvements
        improvements = {}
        for model_name in model_names:
            improvements[model_name] = {}
            baseline = model_results[model_name].get("baseline_metrics", {})
            episodic = model_results[model_name].get("episodic_metrics", {})
            
            for metric in metric_names:
                baseline_val = baseline.get(metric, 0)
                episodic_val = episodic.get(metric, 0)
                
                if baseline_val != 0:
                    improvement = ((episodic_val - baseline_val) / baseline_val) * 100
                else:
                    improvement = 0 if episodic_val == 0 else 100  # If baseline is 0, use 0% or 100% improvement
                
                improvements[model_name][metric] = improvement
        
        # Plot improvement percentages
        x = np.arange(len(metric_names))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            model_improvements = [improvements[model_name].get(metric, 0) for metric in metric_names]
            pos = x + (i - len(model_names)/2 + 0.5) * width
            bars = ax_improvement.bar(pos, model_improvements, width, label=model_name)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax_improvement.annotate(f'{height:.1f}%',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3),  # 3 points vertical offset
                                       textcoords="offset points",
                                       ha='center', va='bottom')
        
        ax_improvement.set_title("Percentage Improvement with Episodic Memory", fontsize=14)
        ax_improvement.set_xticks(x)
        ax_improvement.set_xticklabels(metric_names)
        ax_improvement.legend()
        ax_improvement.set_ylabel("Improvement (%)")
        
        # Memory efficiency comparison
        ax_memory = fig.add_subplot(gs[1, 0])
        
        # Prepare data for memory comparison
        memory_data = {}
        for model_name in model_names:
            baseline = model_results[model_name].get("baseline_metrics", {})
            episodic = model_results[model_name].get("episodic_metrics", {})
            
            memory_data[model_name] = {
                "memory_usage": episodic.get("memory_usage", 0),
                "memory_efficiency": episodic.get("memory_efficiency", 0),
                "context_relevance": episodic.get("context_relevance", 0)
            }
        
        # Plot memory efficiency vs context relevance
        for model_name in model_names:
            x_val = memory_data[model_name]["memory_efficiency"]
            y_val = memory_data[model_name]["context_relevance"]
            size = memory_data[model_name]["memory_usage"] * 100
            ax_memory.scatter(x_val, y_val, s=size, label=model_name, alpha=0.7)
        
        ax_memory.set_title("Memory Efficiency vs Context Relevance", fontsize=14)
        ax_memory.set_xlabel("Memory Efficiency")
        ax_memory.set_ylabel("Context Relevance")
        ax_memory.legend()
        ax_memory.grid(True, linestyle='--', alpha=0.7)
        
        # Overall score comparison
        ax_overall = fig.add_subplot(gs[1, 1])
        
        # Calculate overall scores
        overall_scores = {}
        for model_name in model_names:
            baseline = model_results[model_name].get("baseline_metrics", {})
            episodic = model_results[model_name].get("episodic_metrics", {})
            
            # Get average of key metrics
            baseline_avg = sum(baseline.get(metric, 0) for metric in metric_names) / len(metric_names)
            episodic_avg = sum(episodic.get(metric, 0) for metric in metric_names) / len(metric_names)
            
            overall_scores[model_name] = {
                "baseline": baseline_avg,
                "episodic": episodic_avg
            }
        
        # Plot overall scores
        model_indices = np.arange(len(model_names))
        bar_width = 0.35
        
        baseline_scores = [overall_scores[model]["baseline"] for model in model_names]
        episodic_scores = [overall_scores[model]["episodic"] for model in model_names]
        
        ax_overall.bar(model_indices - bar_width/2, baseline_scores, bar_width, label='Baseline')
        ax_overall.bar(model_indices + bar_width/2, episodic_scores, bar_width, label='Episodic Memory')
        
        ax_overall.set_title("Overall Performance Score", fontsize=14)
        ax_overall.set_xticks(model_indices)
        ax_overall.set_xticklabels(model_names, rotation=45)
        ax_overall.legend()
        ax_overall.set_ylabel("Average Score")
        
        # Add overall title
        plt.suptitle(f"Multi-Model Benchmark Comparison", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"multi_model_comparison_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return output_path

    def _generate_performance_radar_chart(self, output_dir: str) -> str:
        """Generate a radar chart showing model performance across metrics"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            
            # Define metrics to include in radar chart
            metrics = [
                "accuracy",
                "context_relevance",
                "coherence",
                "response_quality",
                "memory_usage",
                "memory_efficiency"
            ]
            
            # Get metric values
            baseline_values = [self.baseline_metrics.get_metric(m, 0) for m in metrics]
            episodic_values = [self.episodic_metrics.get_metric(m, 0) for m in metrics]
            
            # Number of metrics
            N = len(metrics)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add extra element to close the loop
            baseline_values += baseline_values[:1]
            episodic_values += episodic_values[:1]
            
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot baseline
            ax.plot(angles, baseline_values, 'b-', linewidth=1, label='Baseline')
            ax.fill(angles, baseline_values, 'b', alpha=0.1)
            
            # Plot episodic memory
            ax.plot(angles, episodic_values, 'r-', linewidth=1, label='Episodic Memory')
            ax.fill(angles, episodic_values, 'r', alpha=0.1)
            
            # Set labels
            ax.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, metrics)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            ax.set_title(f"Performance Comparison: {self.model_name}", fontsize=14)
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"radar_chart_{self.model_name}_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        
        except Exception as e:
            print(f"Error generating radar chart: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _generate_conversation_coherence_chart(self, output_dir: str) -> str:
        """Generate a chart showing conversation coherence over time"""
        try:
            import matplotlib.pyplot as plt
            
            # Get conversation coherence scores
            baseline_scores = self.baseline_metrics.get_metric_list('conversation_coherence_scores', [])
            episodic_scores = self.episodic_metrics.get_metric_list('conversation_coherence_scores', [])
            
            if not baseline_scores or not episodic_scores:
                print("No conversation coherence scores available for visualization")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot scores
            x = range(1, len(baseline_scores) + 1)
            ax.plot(x, baseline_scores, 'b-', marker='o', label='Baseline')
            ax.plot(x, episodic_scores, 'r-', marker='o', label='Episodic Memory')
            
            # Add reference line for y=0.5
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Conversation Exchange')
            ax.set_ylabel('Coherence Score')
            ax.set_title(f'Conversation Coherence Over Time: {self.model_name}')
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"conversation_coherence_{self.model_name}_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        
        except Exception as e:
            print(f"Error generating conversation coherence chart: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _generate_accuracy_comparison(self, output_dir: str) -> str:
        """
        Generate visualization comparing factual accuracy between baseline and episodic memory.
        
        Args:
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get baseline and episodic metrics for comparison
            baseline_accuracy = [score for score in self.baseline_metrics.factual_accuracy_scores] if hasattr(self.baseline_metrics, 'factual_accuracy_scores') else []
            episodic_accuracy = [score for score in self.episodic_metrics.factual_accuracy_scores] if hasattr(self.episodic_metrics, 'factual_accuracy_scores') else []
            
            # If no factual accuracy scores available, return None
            if not baseline_accuracy and not episodic_accuracy:
                print("No factual accuracy scores available for visualization")
                return None
            
            # Make sure the arrays are the same length
            max_len = max(len(baseline_accuracy), len(episodic_accuracy))
            baseline_accuracy = baseline_accuracy + [0] * (max_len - len(baseline_accuracy))
            episodic_accuracy = episodic_accuracy + [0] * (max_len - len(episodic_accuracy))
            
            # Create identifiers for x-axis (scenario numbers)
            scenario_labels = [f"Scenario {i+1}" for i in range(len(baseline_accuracy))]
            
            # Create subplots
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "bar"}, {"type": "bar"}]],
                               subplot_titles=["Factual Accuracy by Scenario", "Average Factual Accuracy"],
                               column_widths=[0.6, 0.4])
            
            # Add individual scenario comparison bar chart
            fig.add_trace(
                go.Bar(
                    x=scenario_labels,
                    y=baseline_accuracy,
                    name="Baseline",
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=scenario_labels,
                    y=episodic_accuracy,
                    name="Episodic Memory",
                    marker_color='forestgreen'
                ),
                row=1, col=1
            )
            
            # Add average accuracy comparison
            avg_baseline = sum(baseline_accuracy) / len(baseline_accuracy) if baseline_accuracy else 0
            avg_episodic = sum(episodic_accuracy) / len(episodic_accuracy) if episodic_accuracy else 0
            
            fig.add_trace(
                go.Bar(
                    x=["Average Accuracy"],
                    y=[avg_baseline],
                    name="Baseline",
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Average Accuracy"],
                    y=[avg_episodic],
                    name="Episodic Memory",
                    marker_color='forestgreen',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add percentage improvement label
            improvement_pct = ((avg_episodic - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
            
            fig.add_annotation(
                x="Average Accuracy",
                y=max(avg_baseline, avg_episodic) + 0.05,
                text=f"{improvement_pct:.1f}% improvement",
                showarrow=True,
                arrowhead=1,
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Factual Accuracy Comparison: {self.model_name}",
                barmode='group',
                height=500,
                width=1000,
                margin=dict(l=50, r=50, t=80, b=80),
                template="plotly_white",
                yaxis_title="Accuracy Score",
                xaxis_title="Test Scenario",
                yaxis_range=[0, 1.1],
                yaxis2_range=[0, 1.1]
            )
            
            # Save visualization file
            output_file = os.path.join(output_dir, "accuracy_comparison.html")
            fig.write_html(output_file)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating accuracy comparison visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_factual_recall_visualization(self, output_dir: str) -> str:
        """
        Generate visualization comparing factual recall metrics between baseline and episodic memory.
        
        Args:
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get baseline and episodic metrics for comparison
            metrics_to_compare = [
                'factual_question_accuracy_scores',
                'factual_recall_scores',
                'consistency_over_time_scores'
            ]
            
            # Check if we have data for these metrics
            has_metrics = False
            for metric in metrics_to_compare:
                if hasattr(self.baseline_metrics, metric) and getattr(self.baseline_metrics, metric):
                    has_metrics = True
                    break
                    
            if not has_metrics:
                print("No factual recall metrics available for visualization")
                return None
            
            # Create subplots - one row for each metric type
            fig = make_subplots(
                rows=len(metrics_to_compare),
                cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}]] * len(metrics_to_compare),
                subplot_titles=[
                    "Factual Question Accuracy by Scenario", "Average Factual Question Accuracy",
                    "Factual Recall by Scenario", "Average Factual Recall",
                    "Consistency Over Time by Scenario", "Average Consistency Over Time"
                ],
                vertical_spacing=0.1
            )
            
            # Process each metric
            for i, metric in enumerate(metrics_to_compare):
                row = i + 1
                
                # Get baseline and episodic values for this metric
                baseline_values = getattr(self.baseline_metrics, metric, []) if hasattr(self.baseline_metrics, metric) else []
                episodic_values = getattr(self.episodic_metrics, metric, []) if hasattr(self.episodic_metrics, metric) else []
                
                # Skip if no data
                if not baseline_values and not episodic_values:
                    continue
                    
                # Make sure the arrays are the same length
                max_len = max(len(baseline_values), len(episodic_values))
                baseline_values = baseline_values + [0] * (max_len - len(baseline_values))
                episodic_values = episodic_values + [0] * (max_len - len(episodic_values))
                
                # Create scenario labels
                scenario_labels = [f"Scenario {i+1}" for i in range(len(baseline_values))]
                
                # Add individual scenario comparison
                fig.add_trace(
                    go.Bar(
                        x=scenario_labels,
                        y=baseline_values,
                        name="Baseline",
                        marker_color='lightblue',
                        showlegend=i==0
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=scenario_labels,
                        y=episodic_values,
                        name="Episodic Memory",
                        marker_color='forestgreen',
                        showlegend=i==0
                    ),
                    row=row, col=1
                )
                
                # Add average comparison
                avg_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
                avg_episodic = sum(episodic_values) / len(episodic_values) if episodic_values else 0
                
                fig.add_trace(
                    go.Bar(
                        x=["Average"],
                        y=[avg_baseline],
                        name="Baseline",
                        marker_color='lightblue',
                        showlegend=False
                    ),
                    row=row, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=["Average"],
                        y=[avg_episodic],
                        name="Episodic Memory",
                        marker_color='forestgreen',
                        showlegend=False
                    ),
                    row=row, col=2
                )
                
                # Add percentage improvement label
                improvement_pct = ((avg_episodic - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
                
                fig.add_annotation(
                    x="Average",
                    y=max(avg_baseline, avg_episodic) + 0.05,
                    text=f"{improvement_pct:.1f}% improvement",
                    showarrow=True,
                    arrowhead=1,
                    row=row, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Factual Recall Metrics Comparison: {self.model_name}",
                barmode='group',
                height=300 * len(metrics_to_compare),
                width=1000,
                margin=dict(l=50, r=50, t=80, b=50),
                template="plotly_white"
            )
            
            # Save visualization file
            output_file = os.path.join(output_dir, "factual_recall_comparison.html")
            fig.write_html(output_file)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating factual recall comparison visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_memory_usage_visualization(self, output_dir: str) -> str:
        """
        Generate visualization showing memory usage comparison.
        
        Args:
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get memory-related metrics if available
            memory_metrics = {
                "episodic": getattr(self.episodic_metrics, "memory_usage_scores", []),
                "episodic_avg": getattr(self.episodic_metrics, "memory_usage_avg", 0),
                "retrieval_accuracy": getattr(self.episodic_metrics, "memory_retrieval_accuracy_scores", []),
                "retrieval_accuracy_avg": getattr(self.episodic_metrics, "memory_retrieval_accuracy_avg", 0)
            }
            
            # If no memory metrics available, return None
            if not memory_metrics["episodic"] and not memory_metrics["retrieval_accuracy"]:
                print("No memory usage metrics available for visualization")
                return None
            
            # Create subplots
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "bar"}, {"type": "bar"}]],
                               subplot_titles=["Memory Usage by Scenario", "Memory Metrics"],
                               column_widths=[0.5, 0.5])
            
            # Add memory usage by scenario if available
            if memory_metrics["episodic"]:
                scenario_labels = [f"Scenario {i+1}" for i in range(len(memory_metrics["episodic"]))]
                
                fig.add_trace(
                    go.Bar(
                        x=scenario_labels,
                        y=memory_metrics["episodic"],
                        name="Memory Usage",
                        marker_color='purple'
                    ),
                    row=1, col=1
                )
            
            # Add memory metrics comparison
            metrics_to_show = []
            values_to_show = []
            
            if memory_metrics["episodic_avg"]:
                metrics_to_show.append("Avg Memory Usage")
                values_to_show.append(memory_metrics["episodic_avg"])
                
            if memory_metrics["retrieval_accuracy_avg"]:
                metrics_to_show.append("Retrieval Accuracy")
                values_to_show.append(memory_metrics["retrieval_accuracy_avg"])
            
            if metrics_to_show:
                fig.add_trace(
                    go.Bar(
                        x=metrics_to_show,
                        y=values_to_show,
                        marker_color=['purple', 'teal'][:len(metrics_to_show)]
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Memory Usage Metrics: {self.model_name}",
                height=500,
                width=1000,
                margin=dict(l=50, r=50, t=80, b=80),
                template="plotly_white"
            )
            
            # Save visualization file
            output_file = os.path.join(output_dir, "memory_usage.html")
            fig.write_html(output_file)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating memory usage visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_response_quality_visualization(self, output_dir: str) -> str:
        """
        Generate visualization showing response quality comparison.
        
        Args:
            output_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get response quality metrics if available
            response_quality_metrics = {
                "baseline": getattr(self.baseline_metrics, "response_quality_scores", []),
                "episodic": getattr(self.episodic_metrics, "response_quality_scores", []),
                "baseline_avg": getattr(self.baseline_metrics, "response_quality_avg", 0),
                "episodic_avg": getattr(self.episodic_metrics, "response_quality_avg", 0)
            }
            
            # If no response quality metrics available, return None
            if not response_quality_metrics["baseline"] and not response_quality_metrics["episodic"]:
                print("No response quality metrics available for visualization")
                return None
            
            # Create subplots
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "bar"}, {"type": "bar"}]],
                               subplot_titles=["Response Quality by Scenario", "Average Response Quality"],
                               column_widths=[0.5, 0.5])
            
            # Add response quality by scenario if available
            if response_quality_metrics["baseline"]:
                scenario_labels = [f"Scenario {i+1}" for i in range(len(response_quality_metrics["baseline"]))]
                
                fig.add_trace(
                    go.Bar(
                        x=scenario_labels,
                        y=response_quality_metrics["baseline"],
                        name="Baseline",
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Bar(
                    x=scenario_labels,
                    y=response_quality_metrics["episodic"],
                    name="Episodic Memory",
                    marker_color='forestgreen'
                ),
                row=1, col=1
            )
            
            # Add average response quality comparison
            avg_baseline = response_quality_metrics["baseline_avg"]
            avg_episodic = response_quality_metrics["episodic_avg"]
            
            fig.add_trace(
                go.Bar(
                    x=["Average Response Quality"],
                    y=[avg_baseline],
                    name="Baseline",
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=["Average Response Quality"],
                    y=[avg_episodic],
                    name="Episodic Memory",
                    marker_color='forestgreen',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add percentage improvement label
            improvement_pct = ((avg_episodic - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0
            
            fig.add_annotation(
                x="Average Response Quality",
                y=max(avg_baseline, avg_episodic) + 0.05,
                text=f"{improvement_pct:.1f}% improvement",
                showarrow=True,
                arrowhead=1,
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Response Quality Comparison: {self.model_name}",
                barmode='group',
                height=500,
                width=1000,
                margin=dict(l=50, r=50, t=80, b=80),
                template="plotly_white",
                yaxis_title="Response Quality Score",
                xaxis_title="Test Scenario",
                yaxis_range=[0, 1.1],
                yaxis2_range=[0, 1.1]
            )
            
            # Save visualization file
            output_file = os.path.join(output_dir, "response_quality.html")
            fig.write_html(output_file)
            
            return output_file
            
        except Exception as e:
            print(f"Error generating response quality visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

# Enhanced test scenarios inspired by EMENT research
TEST_SCENARIOS = [
    # Basic information retrieval
    {
        'input': 'What programming languages are best for AI development?',
        'context': 'Python is widely used in AI development due to its extensive libraries like TensorFlow and PyTorch.',
        'expected_output': 'Python is the most popular language for AI development, followed by languages like R and Julia.',
        'key_entities': ['Python', 'TensorFlow', 'PyTorch', 'R', 'Julia'],
        'seed_context': 'Python is widely used in AI development due to its extensive libraries like TensorFlow and PyTorch. R is also popular for statistical analysis, while Julia is gaining traction for its performance.'
    },
    
    # Technical information with entities
    {
        'input': 'Tell me about machine learning frameworks.',
        'context': 'TensorFlow and PyTorch are leading deep learning frameworks. TensorFlow was developed by Google, while PyTorch was created by Facebook.',
        'expected_output': 'TensorFlow and PyTorch are popular frameworks for machine learning and deep learning. TensorFlow was developed by Google, and PyTorch was created by Facebook.',
        'key_entities': ['TensorFlow', 'PyTorch', 'Google', 'Facebook'],
        'seed_context': 'TensorFlow and PyTorch are leading deep learning frameworks. TensorFlow was developed by Google, while PyTorch was created by Facebook. Other frameworks include Keras, which is now integrated with TensorFlow, and JAX, which is gaining popularity.'
    },
    
    # Personal information retrieval
    {
        'input': "What's my cat's name and what problem does she have?",
        'context': "My cat Mimi is sick because she loves eating lemons, but they make her sick.",
        'expected_output': "Your cat's name is Mimi. She gets sick because she eats lemons, which aren't good for her.",
        'key_entities': ['Mimi', 'lemons', 'sick'],
        'seed_context': "I have a cat named Mimi. She's a beautiful tabby cat. Unfortunately, Mimi loves eating lemons, but they always make her sick. I've been trying to keep lemons away from her."
    },
    
    # Follow-up question to test temporal consistency
    {
        'input': "What should I do to help my cat?",
        'context': "Keep lemons away from Mimi and consult a veterinarian if she continues to be sick.",
        'expected_output': "You should keep lemons away from Mimi since they make her sick. If she continues to be unwell, you should consult a veterinarian.",
        'key_entities': ['Mimi', 'lemons', 'veterinarian'],
        'is_followup': True,
        'previous_context': "My cat Mimi is sick because she loves eating lemons, but they make her sick."
    },
    
    # Complex information with multiple entities
    {
        'input': "Who are the leaders of Netflix?",
        'context': "Ted Sarandos and Greg Peters are co-CEOs of Netflix, with Reed Hastings as Executive Chairman.",
        'expected_output': "Netflix is led by Ted Sarandos and Greg Peters as co-CEOs, with Reed Hastings serving as Executive Chairman.",
        'key_entities': ['Ted Sarandos', 'Greg Peters', 'Reed Hastings', 'Netflix'],
        'seed_context': "Ted Sarandos and Greg Peters are now co-CEOs of Netflix, with Reed Hastings as Executive Chairman. This leadership structure was announced in 2022 as part of Netflix's succession planning."
    },
    
    # Multi-turn conversation to test coherence
    {
        'input': "What's the capital of France?",
        'context': "Paris is the capital of France.",
        'expected_output': "Paris is the capital of France.",
        'key_entities': ['Paris', 'France'],
                'seed_context': "Paris is the capital of France. It's known as the City of Light and is famous for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
    },
    
    # Follow-up question for the previous context
    {
        'input': "What famous landmarks can I visit there?",
        'context': "In Paris, you can visit the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        'expected_output': "In Paris, you can visit famous landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        'key_entities': ['Eiffel Tower', 'Louvre Museum', 'Notre-Dame Cathedral'],
        'is_followup': True,
        'previous_context': "Paris is the capital of France."
    },
    
    # Long-term memory test
    {
        'input': "What was the name of my cat that I mentioned earlier?",
        'context': "I previously mentioned my cat Mimi who gets sick from eating lemons.",
        'expected_output': "Your cat's name is Mimi, who you mentioned gets sick from eating lemons.",
        'key_entities': ['Mimi', 'lemons'],
        'is_followup': True,
        'previous_context': "My cat Mimi is sick because she loves eating lemons, but they make her sick."
    },
    
    # Technical follow-up to test knowledge retention
    {
        'input': "Which company developed TensorFlow?",
        'context': "TensorFlow was developed by Google.",
        'expected_output': "TensorFlow was developed by Google.",
        'key_entities': ['TensorFlow', 'Google'],
        'is_followup': True,
        'previous_context': "TensorFlow and PyTorch are leading deep learning frameworks. TensorFlow was developed by Google, while PyTorch was created by Facebook."
    },
    
    # Complex reasoning with multiple entities
    {
        'input': "Compare the leadership structures of tech companies we've discussed.",
        'context': "We discussed Netflix's leadership with Ted Sarandos and Greg Peters as co-CEOs and Reed Hastings as Executive Chairman. We also mentioned Google as the developer of TensorFlow and Facebook as the creator of PyTorch.",
        'expected_output': "We've discussed Netflix, which has Ted Sarandos and Greg Peters as co-CEOs with Reed Hastings as Executive Chairman. We also mentioned Google as the developer of TensorFlow and Facebook as the creator of PyTorch, but didn't specifically discuss their leadership structures.",
        'key_entities': ['Netflix', 'Ted Sarandos', 'Greg Peters', 'Reed Hastings', 'Google', 'Facebook'],
        'is_followup': True,
        'previous_context': "Ted Sarandos and Greg Peters are co-CEOs of Netflix, with Reed Hastings as Executive Chairman. TensorFlow was developed by Google, while PyTorch was created by Facebook."
    }
]

# Function to create model-specific test scenarios
def create_model_specific_scenarios(model_name: str) -> List[Dict[str, Any]]:
    """Create test scenarios optimized for specific models"""
    # Base scenarios are good for all models
    scenarios = TEST_SCENARIOS.copy()
    
    # For larger models, add more complex scenarios
    if "70b" in model_name or "90b" in model_name:
        complex_scenarios = [
            {
                'input': "Explain the differences between supervised and unsupervised learning in machine learning.",
                'context': "Supervised learning uses labeled data with input-output pairs, while unsupervised learning finds patterns in unlabeled data. Semi-supervised learning uses both labeled and unlabeled data.",
                'expected_output': "Supervised learning uses labeled data where the algorithm learns from input-output pairs. Unsupervised learning works with unlabeled data to find patterns or structure. Semi-supervised learning combines both approaches by using a small amount of labeled data with a larger amount of unlabeled data.",
                'key_entities': ['supervised learning', 'unsupervised learning', 'semi-supervised learning', 'labeled data', 'unlabeled data'],
                'seed_context': "In machine learning, supervised learning uses labeled data with input-output pairs for training, while unsupervised learning finds patterns in unlabeled data without explicit guidance. Semi-supervised learning combines both approaches by using a small amount of labeled data with a larger amount of unlabeled data."
            },
            {
                'input': "What are the key components of a transformer architecture in NLP?",
                'context': "Transformer architecture includes self-attention mechanisms, positional encoding, feed-forward networks, and layer normalization. It was introduced in the 'Attention is All You Need' paper.",
                'expected_output': "The key components of a transformer architecture include self-attention mechanisms, positional encoding, feed-forward neural networks, and layer normalization. This architecture was introduced in the 'Attention is All You Need' paper and forms the foundation of models like BERT and GPT.",
                'key_entities': ['transformer', 'self-attention', 'positional encoding', 'feed-forward networks', 'layer normalization', 'BERT', 'GPT'],
                'seed_context': "The transformer architecture, introduced in the 'Attention is All You Need' paper, includes several key components: self-attention mechanisms that allow the model to weigh the importance of different words in a sequence, positional encoding to maintain word order information, feed-forward neural networks for processing, and layer normalization for training stability. This architecture forms the foundation of modern NLP models like BERT and GPT."
            }
        ]
        scenarios.extend(complex_scenarios)
    
    return scenarios