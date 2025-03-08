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
    def __init__(self):
        self.response_times = []
        self.memory_accuracies = []
        self.context_scores = []
        self.token_counts = []
        self.relevance_scores = []
        self.entity_recall_scores = []
        self.temporal_consistency_scores = []
        self.factual_accuracy_scores = []
        self.conversation_coherence_scores = []
        
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

class BenchmarkingSystem:
    def __init__(self, llm_provider, test_scenarios: List[Dict[str, Any]], model_name: str = "llama3-8b-8192"):
        self.llm_provider = llm_provider
        self.test_scenarios = test_scenarios
        self.model_name = model_name
        self.traditional_metrics = BenchmarkMetrics()
        self.episodic_metrics = BenchmarkMetrics()
        self.conversation_history = {}
        
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
                self.traditional_metrics.add_metric('response_times', end_time - start_time)
                self.traditional_metrics.add_metric('relevance_scores', 
                    self._calculate_relevance(response, scenario['expected_output']))
                self.traditional_metrics.add_metric('context_scores',
                    self._calculate_context_preservation(response, scenario['context']))
                
                # Calculate entity recall
                if 'key_entities' in scenario:
                    self.traditional_metrics.add_metric('entity_recall_scores',
                        self._calculate_entity_recall(response, scenario.get('key_entities', [])))
                
                # Calculate temporal consistency for follow-up questions
                if scenario.get('is_followup', False) and 'previous_context' in scenario:
                    self.traditional_metrics.add_metric('temporal_consistency_scores',
                        self._calculate_temporal_consistency(response, scenario.get('previous_context', '')))
                
                # Calculate factual accuracy
                self.traditional_metrics.add_metric('factual_accuracy_scores',
                    self._calculate_factual_accuracy(response, scenario['expected_output']))
                
                # Calculate conversation coherence
                if len(self.conversation_history[scenario_idx]) >= 4:  # At least 2 exchanges
                    self.traditional_metrics.add_metric('conversation_coherence_scores',
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
        """Calculate context preservation score"""
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        if not context_words:
            return 0.0
        return len(response_words.intersection(context_words)) / len(context_words)

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
        """Calculate entity recall score"""
        if not key_entities:
            return 0.0
        
        response_lower = response.lower()
        found_entities = sum(1 for entity in key_entities if entity.lower() in response_lower)
        return found_entities / len(key_entities)
    
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
        # more sophisticated NLP techniques or even another LLM to evaluate
        response_lower = response.lower()
        expected_lower = expected_output.lower()
        
        # Check for key phrases or facts
        key_phrases = expected_lower.split('. ')
        if not key_phrases:
            return 0.0
        
        found_phrases = sum(1 for phrase in key_phrases if phrase and phrase in response_lower)
        return found_phrases / len(key_phrases)
    
    def _calculate_conversation_coherence(self, conversation: List[Dict]) -> float:
        """Calculate conversation coherence score"""
        # This is a simplified version - in a real system, you might use
        # more sophisticated techniques
        coherence_score = 0.0
        
        # Check if responses reference previous exchanges
        for i in range(3, len(conversation), 2):  # Check assistant responses
            current_response = conversation[i]['content'].lower()
            previous_user = conversation[i-1]['content'].lower()
            previous_assistant = conversation[i-2]['content'].lower()
            
            # Check if current response references previous exchanges
            user_words = set(previous_user.split())
            assistant_words = set(previous_assistant.split())
            current_words = set(current_response.split())
            
            user_overlap = len(current_words.intersection(user_words)) / len(user_words) if user_words else 0
            assistant_overlap = len(current_words.intersection(assistant_words)) / len(assistant_words) if assistant_words else 0
            
            coherence_score += (user_overlap + assistant_overlap) / 2
        
        # Average coherence score
        num_checks = max(1, (len(conversation) - 3) // 2)
        return coherence_score / num_checks

    def generate_visualizations(self, save_dir: str = 'visualizations'):
        """Generate and save benchmark visualizations"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Check if we have enough data to generate visualizations
            if not self.traditional_metrics.response_times or not self.episodic_metrics.response_times:
                print("Warning: Not enough data to generate visualizations")
                self._generate_error_visualization(save_dir)
                return
            
            # 1. Comprehensive Metrics Comparison
            self._generate_comprehensive_comparison(save_dir)
            
            # 2. Memory Accuracy Visualization
            if self.episodic_metrics.memory_accuracies:
                self._generate_memory_accuracy_viz(save_dir)
            
            # 3. Response Time Comparison
            self._generate_response_time_viz(save_dir)
            
            # 4. Entity Recall Comparison
            if self.traditional_metrics.entity_recall_scores and self.episodic_metrics.entity_recall_scores:
                self._generate_entity_recall_viz(save_dir)
            
            # 5. Temporal Consistency Visualization
            if self.traditional_metrics.temporal_consistency_scores and self.episodic_metrics.temporal_consistency_scores:
                self._generate_temporal_consistency_viz(save_dir)
            
            # 6. Conversation Coherence Visualization
            if self.traditional_metrics.conversation_coherence_scores and self.episodic_metrics.conversation_coherence_scores:
                self._generate_conversation_coherence_viz(save_dir)
            
            # 7. Overall Performance Radar Chart
            self._generate_radar_chart(save_dir)
            
            # 8. Performance Improvement Summary
            self._generate_improvement_summary(save_dir)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            self._generate_error_visualization(save_dir)
    
    def _generate_error_visualization(self, save_dir: str):
        """Generate a simple error visualization when data is insufficient"""
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Benchmarking Error</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1, h2 {
                    color: #333;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                }
                .error-card {
                    background-color: #f8d7da;
                    color: #721c24;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Benchmarking Error</h1>
                <div class="error-card">
                    <h2>Insufficient Data</h2>
                    <p>There was not enough data collected during the benchmarking process to generate visualizations.</p>
                    <p>This could be due to errors during the benchmarking process or insufficient test scenarios.</p>
                    <p>Please check the console output for more details on any errors that occurred.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, "error.html"), "w") as f:
            f.write(error_html)

    def _generate_comprehensive_comparison(self, save_dir: str):
        """Generate comprehensive metrics comparison visualization"""
        # Prepare comparison data
        metrics = [
            'Context Preservation', 
            'Relevance', 
            'Entity Recall', 
            'Factual Accuracy',
            'Memory Accuracy'
        ]
        
        traditional_values = [
            self.traditional_metrics.get_average('context_scores'),
            self.traditional_metrics.get_average('relevance_scores'),
            self.traditional_metrics.get_average('entity_recall_scores'),
            self.traditional_metrics.get_average('factual_accuracy_scores'),
            0  # Traditional doesn't have memory accuracy
        ]
        
        episodic_values = [
            self.episodic_metrics.get_average('context_scores'),
            self.episodic_metrics.get_average('relevance_scores'),
            self.episodic_metrics.get_average('entity_recall_scores'),
            self.episodic_metrics.get_average('factual_accuracy_scores'),
            self.episodic_metrics.get_average('memory_accuracies')
        ]
        
        df = pd.DataFrame({
            'Metric': metrics,
            'Traditional LLM': traditional_values,
            'Episodic Memory': episodic_values
        })
        
        # Calculate improvement percentages
        improvements = []
        for i in range(len(metrics)):
            if traditional_values[i] > 0:
                imp = ((episodic_values[i] - traditional_values[i]) / traditional_values[i]) * 100
                improvements.append(f"+{imp:.1f}%" if imp > 0 else f"{imp:.1f}%")
            else:
                improvements.append("N/A")
        
        df['Improvement'] = improvements
        
        # Create bar chart
        fig = px.bar(
            df, 
            x='Metric', 
            y=['Traditional LLM', 'Episodic Memory'],
            barmode='group',
            title=f'Performance Comparison: Traditional vs Episodic Memory ({self.model_name})',
            labels={'value': 'Score (0-1)', 'variable': 'System Type'},
            color_discrete_sequence=['#636EFA', '#EF553B']
        )
        
        # Add text annotations for improvement percentages
        for i, metric in enumerate(metrics):
            if improvements[i] != "N/A" and "+" in improvements[i]:
                fig.add_annotation(
                    x=metric,
                    y=episodic_values[i] + 0.05,
                    text=improvements[i],
                    showarrow=False,
                    font=dict(color="green", size=12)
                )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_range=[0, 1.1]
        )
        
        fig.write_html(f"{save_dir}/comprehensive_comparison.html")
        
    def _generate_memory_accuracy_viz(self, save_dir: str):
        """Generate memory accuracy visualization"""
        memory_df = pd.DataFrame({
            'Scenario': [f"Scenario {i+1}" for i in range(len(self.episodic_metrics.memory_accuracies))],
            'Memory Accuracy': self.episodic_metrics.memory_accuracies
        })
        
        fig = px.line(
            memory_df, 
            x='Scenario', 
            y='Memory Accuracy',
            title='Memory Accuracy Across Scenarios',
            markers=True,
            labels={'Memory Accuracy': 'Accuracy Score (0-1)'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=memory_df['Scenario']),
            yaxis_range=[0, 1]
        )
        
        fig.write_html(f"{save_dir}/memory_accuracy.html")
        
    def _generate_response_time_viz(self, save_dir: str):
        """Generate response time comparison visualization"""
        response_df = pd.DataFrame({
            'Scenario': [f"Scenario {i+1}" for i in range(len(self.traditional_metrics.response_times))],
            'Traditional LLM': self.traditional_metrics.response_times,
            'Episodic Memory': self.episodic_metrics.response_times
        })
        
        fig = px.line(
            response_df, 
            x='Scenario', 
            y=['Traditional LLM', 'Episodic Memory'],
            title='Response Time Comparison',
            markers=True,
            labels={'value': 'Response Time (seconds)', 'variable': 'System Type'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=response_df['Scenario']),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.write_html(f"{save_dir}/response_time_comparison.html")
        
    def _generate_entity_recall_viz(self, save_dir: str):
        """Generate entity recall comparison visualization"""
        if not self.traditional_metrics.entity_recall_scores:
            return
            
        entity_df = pd.DataFrame({
            'Scenario': [f"Scenario {i+1}" for i in range(len(self.traditional_metrics.entity_recall_scores))],
            'Traditional LLM': self.traditional_metrics.entity_recall_scores,
            'Episodic Memory': self.episodic_metrics.entity_recall_scores
        })
        
        fig = px.bar(
            entity_df, 
            x='Scenario', 
            y=['Traditional LLM', 'Episodic Memory'],
            barmode='group',
            title='Entity Recall Comparison',
            labels={'value': 'Entity Recall Score (0-1)', 'variable': 'System Type'}
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_range=[0, 1]
        )
        
        fig.write_html(f"{save_dir}/entity_recall_comparison.html")
        
    def _generate_temporal_consistency_viz(self, save_dir: str):
        """Generate temporal consistency visualization"""
        if not self.traditional_metrics.temporal_consistency_scores:
            return
            
        temporal_df = pd.DataFrame({
            'Scenario': [f"Scenario {i+1}" for i in range(len(self.traditional_metrics.temporal_consistency_scores))],
            'Traditional LLM': self.traditional_metrics.temporal_consistency_scores,
            'Episodic Memory': self.episodic_metrics.temporal_consistency_scores
        })
        
        fig = px.bar(
            temporal_df, 
            x='Scenario', 
            y=['Traditional LLM', 'Episodic Memory'],
            barmode='group',
            title='Temporal Consistency Comparison',
            labels={'value': 'Consistency Score (0-1)', 'variable': 'System Type'}
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_range=[0, 1]
        )
        
        fig.write_html(f"{save_dir}/temporal_consistency_comparison.html")
        
    def _generate_conversation_coherence_viz(self, save_dir: str):
        """Generate conversation coherence visualization"""
        if not self.traditional_metrics.conversation_coherence_scores:
            return
            
        coherence_df = pd.DataFrame({
            'Scenario': [f"Scenario {i+1}" for i in range(len(self.traditional_metrics.conversation_coherence_scores))],
            'Traditional LLM': self.traditional_metrics.conversation_coherence_scores,
            'Episodic Memory': self.episodic_metrics.conversation_coherence_scores
        })
        
        fig = px.bar(
            coherence_df, 
            x='Scenario', 
            y=['Traditional LLM', 'Episodic Memory'],
            barmode='group',
            title='Conversation Coherence Comparison',
            labels={'value': 'Coherence Score (0-1)', 'variable': 'System Type'}
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_range=[0, 1]
        )
        
        fig.write_html(f"{save_dir}/conversation_coherence_comparison.html")
        
    def _generate_radar_chart(self, save_dir: str):
        """Generate radar chart for overall performance comparison"""
        # Prepare data for radar chart
        metrics = [
            'Context Preservation', 
            'Relevance', 
            'Entity Recall', 
            'Factual Accuracy',
            'Temporal Consistency',
            'Conversation Coherence'
        ]
        
        traditional_values = [
            self.traditional_metrics.get_average('context_scores'),
            self.traditional_metrics.get_average('relevance_scores'),
            self.traditional_metrics.get_average('entity_recall_scores'),
            self.traditional_metrics.get_average('factual_accuracy_scores'),
            self.traditional_metrics.get_average('temporal_consistency_scores') if self.traditional_metrics.temporal_consistency_scores else 0,
            self.traditional_metrics.get_average('conversation_coherence_scores') if self.traditional_metrics.conversation_coherence_scores else 0
        ]
        
        episodic_values = [
            self.episodic_metrics.get_average('context_scores'),
            self.episodic_metrics.get_average('relevance_scores'),
            self.episodic_metrics.get_average('entity_recall_scores'),
            self.episodic_metrics.get_average('factual_accuracy_scores'),
            self.episodic_metrics.get_average('temporal_consistency_scores') if self.episodic_metrics.temporal_consistency_scores else 0,
            self.episodic_metrics.get_average('conversation_coherence_scores') if self.episodic_metrics.conversation_coherence_scores else 0
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=traditional_values,
            theta=metrics,
            fill='toself',
            name='Traditional LLM',
            line_color='#636EFA'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=episodic_values,
            theta=metrics,
            fill='toself',
            name='Episodic Memory',
            line_color='#EF553B'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f'System Capabilities Comparison ({self.model_name})'
        )
        
        fig.write_html(f"{save_dir}/capabilities_radar.html")
        
    def _generate_improvement_summary(self, save_dir: str):
        """Generate improvement summary visualization"""
        # Calculate improvement percentages for each metric
        metrics = [
            'Context Preservation', 
            'Relevance', 
            'Entity Recall', 
            'Factual Accuracy',
            'Temporal Consistency',
            'Conversation Coherence'
        ]
        
        metric_keys = [
            'context_scores',
            'relevance_scores',
            'entity_recall_scores',
            'factual_accuracy_scores',
            'temporal_consistency_scores',
            'conversation_coherence_scores'
        ]
        
        improvements = []
        for key in metric_keys:
            trad_avg = self.traditional_metrics.get_average(key)
            epis_avg = self.episodic_metrics.get_average(key)
            
            if trad_avg > 0:
                imp = ((epis_avg - trad_avg) / trad_avg) * 100
                improvements.append(imp)
            else:
                improvements.append(0)
        
        # Create improvement summary bar chart
        improvement_df = pd.DataFrame({
            'Metric': metrics,
            'Improvement (%)': improvements
        })
        
        fig = px.bar(
            improvement_df, 
            x='Metric', 
            y='Improvement (%)',
            title='Episodic Memory Improvement Over Traditional LLM',
            color='Improvement (%)',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[-10, 50]
        )
        
        fig.update_layout(
            yaxis_title='Improvement (%)'
        )
        
        fig.write_html(f"{save_dir}/improvement_summary.html")

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