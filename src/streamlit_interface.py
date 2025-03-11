import os
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import asyncio
from datetime import datetime
from groq_conversational_bot import main as groq_bot_main
from episodic_memory import EpisodicMemoryModule, MemoryPriority
from llm_providers import GroqProvider
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

load_dotenv()

# Initialize event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

async def init_memory_module():
    """Initialize the memory module and conversation chain asynchronously"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY environment variable is not set!")
        st.stop()
    
    # Initialize Groq chat
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768"  # Using Mixtral instead of llama3
    )
    
    # Initialize conversation components
    system_prompt = 'You are a friendly conversational chatbot powered by Groq. You are helpful, creative, clever, and very friendly.'
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    
    # Create conversation chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    
    # Initialize memory module
    groq_provider = GroqProvider(api_key=api_key, model_name="mixtral-8x7b-32768")
    memory_module = EpisodicMemoryModule(
        llm_provider=groq_provider,
        collection_name="episodic_memory",
        embedding_model="all-MiniLM-L6-v2",
    )
    
    return memory_module, conversation

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory_module' not in st.session_state or 'conversation' not in st.session_state:
    # Initialize both memory module and conversation chain
    st.session_state.memory_module, st.session_state.conversation = loop.run_until_complete(init_memory_module())

def create_knowledge_graph():
    """Create and display knowledge graph visualization using Plotly"""
    try:
        relationships = st.session_state.memory_module.get_entity_relationships()
        
        if not relationships:
            return go.Figure()  # Return empty figure if no relationships
        
        G = nx.Graph()
        edge_weights = []
        node_types = {}
        
        # Add nodes and edges
        for rel in relationships:
            entity1, entity2 = rel['entity1'], rel['entity2']
            G.add_node(entity1)
            G.add_node(entity2)
            weight = rel['relationship_strength']
            G.add_edge(entity1, entity2, weight=weight)
            edge_weights.append(weight)
            
            # Store node types
            node_types[entity1] = rel['entity1_type']
            node_types[entity2] = rel['entity2_type']
        
        if not G.nodes():  # Check if graph is empty
            return go.Figure()
        
        # Use force-directed layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

        # Create node trace
        node_trace = go.Scatter(
            x=[], y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Add node positions and text
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (f"{node}<br>Type: {node_types.get(node, 'Unknown')}",)

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        return go.Figure()  # Return empty figure on error

def create_memory_stats_chart():
    """Create memory statistics visualization"""
    try:
        stats = st.session_state.memory_module.get_memory_stats()
        
        # Check if age_distribution exists, otherwise create default values
        if 'age_distribution' in stats:
            age_dist = stats['age_distribution']
            labels = ['Today', 'This Week', 'This Month', 'Older']
            values = [age_dist.get('today', 0), age_dist.get('this_week', 0), 
                     age_dist.get('this_month', 0), age_dist.get('older', 0)]
        else:
            # Use categories as a fallback
            categories = stats.get('categories', {})
            labels = list(categories.keys())
            values = list(categories.values())
            
            # If no categories, use entity types
            if not labels:
                entity_types = stats.get('entity_types', {})
                labels = list(entity_types.keys())
                values = list(entity_types.values())
                
            # If still no data, use default
            if not labels:
                labels = ['No Data']
                values = [1]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                    hole=.3,
                                    title="Memory Distribution")])
        return fig
    except Exception as e:
        st.error(f"Error creating memory stats chart: {str(e)}")
        return go.Figure()  # Return empty figure on error

async def process_user_input(user_input: str):
    """Process user input through both conversation chain and memory module"""
    try:
        # Get conversational response
        response = st.session_state.conversation.predict(human_input=user_input)
        
        # Store memory
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "category": "conversation",
            "conversation_id": "main_chat"
        }
        
        memory_id = await st.session_state.memory_module.store_memory(
            text=user_input,
            metadata=metadata,
            priority=MemoryPriority.MEDIUM
        )
        
        # Get related memories with a minimum relevance threshold
        related_memories = await st.session_state.memory_module.recall_memories(
            query=user_input,
            top_k=3
        )
        
        # Filter memories by relevance threshold
        relevant_memories = [
            mem for mem in related_memories 
            if mem.get('relevance', 0) > 0.7  # Only keep memories with >70% relevance
        ]
        
        return response, memory_id, relevant_memories
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return "I apologize, but I encountered an error processing your message.", None, []

def main():
    st.title("🤖 Groq Chat with Episodic Memory")
    
    try:
        # Sidebar for visualizations and stats
        with st.sidebar:
            st.subheader("📊 Memory Analytics")
            
            # Memory Statistics
            stats = st.session_state.memory_module.get_memory_stats()
            st.metric("Total Memories", stats['total_memories'])
            st.metric("Average Importance", f"{stats['avg_importance']:.2f}")
            
            # Memory Age Distribution
            st.plotly_chart(create_memory_stats_chart(), use_container_width=True)
            
            # Knowledge Graph
            st.subheader("🕸️ Knowledge Graph")
            st.plotly_chart(create_knowledge_graph(), use_container_width=True)
        
        # Main chat interface
        st.subheader("💭 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What's on your mind?"):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Process input and get responses
            response, memory_id, related_memories = loop.run_until_complete(process_user_input(prompt))
            
            # Display assistant response
            with st.chat_message("assistant"):
                # Display the conversational response
                st.write(response)
                
                # Display related memories in an expander
                if related_memories:
                    with st.expander("📚 Related Memories"):
                        st.write("Here are some relevant memories from our previous conversations:")
                        if len(related_memories) > 0:
                            for mem in related_memories:
                                relevance = mem.get('relevance', 0)
                                st.markdown(f"""
                                - {mem['text']}
                                *(Relevance: {relevance:.2f})*
                                """)
                        else:
                            st.write("No highly relevant memories found for this conversation.")
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Show technical details in an expander
            with st.expander("🔧 Technical Details"):
                st.json({
                    "memory_id": memory_id,
                    "model": "mixtral-8x7b-32768",
                    "memory_count": len(related_memories),
                    "related_memories": related_memories
                })
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 