import streamlit as st
import asyncio
from episodic_memory import EpisodicMemoryModule, MemoryPriority, MemoryConfig
from datetime import datetime, timedelta
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import json
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from chatbot import SmartChatBot
from llm_providers.groq_provider import GroqProvider

# Define comprehensive entity configuration from chat_demo.py
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

# Create a custom initialization function for EpisodicMemoryModule
async def initialize_memory_module():
    # Initialize Groq provider
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable is not set")
        st.stop()
    
    model_name = os.environ.get('MODEL_NAME', 'llama3-8b-8192')
    llm_provider = GroqProvider(api_key=groq_api_key, model_name=model_name)
    
    # Create a custom memory config with maintenance tasks disabled
    memory_config = MemoryConfig(
        max_context_items=10,
        memory_decay_factor=0.95,
        importance_threshold=0.5,
        min_references_to_keep=2,
        # Set intervals to very large values to effectively disable automatic tasks
        decay_interval=999999,  # hours
        cleanup_interval=999999,  # hours
        decay_check_interval=999999  # seconds
    )
    
    # Initialize memory module with enhanced configuration
    memory_module = EpisodicMemoryModule(
        llm_provider=llm_provider,
        collection_name="chatbot_memory",
        embedding_model="all-MiniLM-L6-v2",
        config=memory_config,
        entity_config={
            "config": ENTITY_CONFIG,
            "tech_patterns": TECH_PATTERNS
        }
    )
    
    return memory_module, llm_provider

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = nx.Graph()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None

async def update_knowledge_graph():
    """Update the knowledge graph with current entity relationships"""
    # Get entity relationships (non-async method)
    relationships = st.session_state.memory_llm.get_entity_relationships()
    G = nx.Graph()
    
    for rel in relationships:
        entity1 = rel['entity1']
        entity2 = rel['entity2']
        strength = rel['relationship_strength']
        
        G.add_node(entity1)
        G.add_node(entity2)
        G.add_edge(entity1, entity2, weight=strength)
    
    st.session_state.knowledge_graph = G
    st.session_state.last_update = datetime.now()

def plot_knowledge_graph():
    """Create an interactive plot of the knowledge graph using Plotly"""
    G = st.session_state.knowledge_graph
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    # Create traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ))

    # Color nodes by number of connections
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

async def handle_special_commands(command: str):
    """Handle special chat commands"""
    if command.lower() == '/summary':
        summary = await st.session_state.chatbot.get_conversation_summary()
        return True, f"**Conversation Summary:**\n\n{summary}"
        
    if command.lower() == '/conversations':
        conversations = await st.session_state.chatbot.list_conversations()
        if conversations:
            conv_list = "\n".join([f"- {conv_id}" for conv_id in conversations])
            return True, f"**Available conversations:**\n\n{conv_list}\n\n**Current conversation:** {st.session_state.current_conversation_id}"
        return True, "No conversations available."
        
    if command.lower().startswith('/switch '):
        new_conv_id = command.split(' ')[1]
        await st.session_state.chatbot.switch_conversation(new_conv_id)
        st.session_state.current_conversation_id = new_conv_id
        # Initialize chat history for this conversation if it doesn't exist
        if new_conv_id not in st.session_state.chat_history:
            st.session_state.chat_history[new_conv_id] = []
        return True, f"Switched to conversation: {new_conv_id}"
    
    if command.lower() == '/new':
        new_conv_id = await st.session_state.chatbot.start_conversation()
        st.session_state.current_conversation_id = st.session_state.chatbot.conversation_id
        # Initialize chat history for this conversation
        st.session_state.chat_history[st.session_state.current_conversation_id] = []
        return True, f"Started new conversation: {st.session_state.current_conversation_id}"
        
    return False, ""

async def main():
    st.title("AI Chat & Memory Dashboard")
    
    # Initialize components if not already done
    if not st.session_state.initialized:
        with st.spinner("Initializing memory module and chatbot..."):
            # Initialize memory module and LLM provider
            memory_module, llm_provider = await initialize_memory_module()
            st.session_state.memory_llm = memory_module
            
            # Initialize chatbot
            st.session_state.chatbot = SmartChatBot(
                llm_provider=llm_provider,
                memory_module=memory_module,
                persona="helpful AI assistant with expertise in technology and programming",
                max_context_length=10,
                conversation_id=st.session_state.current_conversation_id
            )
            
            st.session_state.initialized = True
    
    # Create tabs for Chat and Memory Management
    tab1, tab2 = st.tabs(["Chat", "Memory Management"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat")
        
        # Sidebar for conversation management
        with st.sidebar:
            st.subheader("Conversation Management")
            
            # Display available conversations
            st.write("**Special Commands:**")
            st.write("- `/new`: Start a new conversation")
            st.write("- `/summary`: Get conversation summary")
            st.write("- `/conversations`: List available conversations")
            st.write("- `/switch <id>`: Switch to a different conversation")
            
            # Button to start a new conversation
            if st.button("New Conversation"):
                await st.session_state.chatbot.start_conversation()
                st.session_state.current_conversation_id = st.session_state.chatbot.conversation_id
                # Initialize chat history for this conversation
                st.session_state.chat_history[st.session_state.current_conversation_id] = []
                st.rerun()
            
            # List and select conversations
            conversations = await st.session_state.chatbot.list_conversations()
            if conversations:
                st.write("**Available Conversations:**")
                selected_conv = st.selectbox(
                    "Select a conversation",
                    options=conversations,
                    index=conversations.index(st.session_state.current_conversation_id) if st.session_state.current_conversation_id in conversations else 0
                )
                
                if st.button("Switch to Selected"):
                    await st.session_state.chatbot.switch_conversation(selected_conv)
                    st.session_state.current_conversation_id = selected_conv
                    # Initialize chat history for this conversation if it doesn't exist
                    if selected_conv not in st.session_state.chat_history:
                        st.session_state.chat_history[selected_conv] = []
                    st.rerun()
        
        # Initialize conversation if none exists
        if not st.session_state.current_conversation_id:
            await st.session_state.chatbot.start_conversation()
            st.session_state.current_conversation_id = st.session_state.chatbot.conversation_id
            st.session_state.chat_history[st.session_state.current_conversation_id] = []
        
        # Display current conversation ID
        st.info(f"Current conversation: {st.session_state.current_conversation_id}")
        
        # Display chat history for current conversation
        if st.session_state.current_conversation_id in st.session_state.chat_history:
            for message in st.session_state.chat_history[st.session_state.current_conversation_id]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Check for special commands
            is_command, command_response = await handle_special_commands(prompt)
            
            if is_command:
                # Add system message to chat history
                if st.session_state.current_conversation_id not in st.session_state.chat_history:
                    st.session_state.chat_history[st.session_state.current_conversation_id] = []
                
                st.session_state.chat_history[st.session_state.current_conversation_id].append({
                    "role": "assistant", 
                    "content": command_response
                })
                
                # Display system message
                with st.chat_message("assistant"):
                    st.write(command_response)
                
                st.rerun()
            else:
                # Add user message to chat history
                if st.session_state.current_conversation_id not in st.session_state.chat_history:
                    st.session_state.chat_history[st.session_state.current_conversation_id] = []
                
                st.session_state.chat_history[st.session_state.current_conversation_id].append({
                    "role": "user", 
                    "content": prompt
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = await st.session_state.chatbot.chat(prompt)
                        st.write(response)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history[st.session_state.current_conversation_id].append({
                            "role": "assistant", 
                            "content": response
                        })
                
                # Update knowledge graph
                await update_knowledge_graph()
    
    # Tab 2: Memory Management
    with tab2:
        # Add auto-refresh functionality
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        if auto_refresh:
            st.sidebar.info(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Memory management tabs
        mem_tab1, mem_tab2, mem_tab3, mem_tab4 = st.tabs([
            "Memory Search", "Knowledge Graph", "Statistics", "Entity Analysis"
        ])
        
        # Tab 2.1: Memory Search
        with mem_tab1:
            st.header("Search Memories")
            search_query = st.text_input("Search Query")
            search_category = st.text_input("Category Filter (optional)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                days_ago = st.number_input("Days to Look Back", min_value=1, value=7)
            with col2:
                max_results = st.number_input("Max Results", min_value=1, value=5)
            with col3:
                # Get conversations (async method)
                conversations = await st.session_state.chatbot.list_conversations()
                conversation_filter = st.selectbox(
                    "Filter by Conversation",
                    options=["All"] + (conversations or []),
                    index=0
                )
                
            if st.button("Search"):
                with st.spinner("Searching..."):
                    start_time = datetime.now() - timedelta(days=days_ago)
                    
                    # Build filters
                    filters = {}
                    if search_category:
                        filters["category"] = search_category
                    if conversation_filter != "All":
                        filters["conversation_id"] = conversation_filter
                    
                    # Use recall_memories (async method) instead of recall_memories_advanced
                    # which might not exist in the current implementation
                    results = await st.session_state.memory_llm.recall_memories(
                        query=search_query,
                        category=search_category if search_category else None,
                        conversation_id=conversation_filter if conversation_filter != "All" else None,
                        min_importance=0.0,
                        top_k=max_results
                    )
                    
                    if results:
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"Result {idx}: {result['text'][:100]}..."):
                                st.write(result)
                    else:
                        st.info("No results found")
        
        # Tab 2.2: Knowledge Graph
        with mem_tab2:
            st.header("Knowledge Graph")
            if st.button("Update Knowledge Graph"):
                with st.spinner("Updating knowledge graph..."):
                    await update_knowledge_graph()
                    st.success("Knowledge graph updated")
            
            st.plotly_chart(plot_knowledge_graph(), use_container_width=True)
        
        # Tab 2.3: Statistics
        with mem_tab3:
            st.header("Memory Statistics")
            if st.button("Get Memory Statistics"):
                with st.spinner("Fetching memory statistics..."):
                    # Use get_memory_stats (non-async method)
                    stats = st.session_state.memory_llm.get_memory_stats()
                    
                    if stats:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Memories", stats.get("memory_count", 0))
                            st.metric("Average Importance", round(stats.get("avg_importance", 0), 2))
                        with col2:
                            st.write("Categories:", ", ".join(stats.get("categories", [])))
                            
                        # Age distribution chart
                        age_dist = stats.get("age_distribution", {})
                        if age_dist:
                            st.bar_chart(age_dist)
                    else:
                        st.info("No memory statistics available")
                    
            # Conversation statistics
            st.subheader("Conversation Statistics")
            if st.button("Get Conversation Statistics"):
                with st.spinner("Fetching conversation statistics..."):
                    # Get conversations (async method)
                    conversations = await st.session_state.chatbot.list_conversations()
                    
                    if conversations:
                        conv_data = []
                        for conv_id in conversations:
                            # Get memories for this conversation (non-async method)
                            with st.session_state.memory_llm.driver.session() as session:
                                conv_memories = session.execute_read(
                                    st.session_state.memory_llm._get_conversation_memories_tx,
                                    conversation_id=conv_id
                                )
                            
                            # Count messages
                            user_msgs = sum(1 for mem in conv_memories if mem.get('metadata', {}).get('type') == 'user_input')
                            assistant_msgs = sum(1 for mem in conv_memories if mem.get('metadata', {}).get('type') == 'assistant_response')
                            
                            conv_data.append({
                                "Conversation": conv_id,
                                "User Messages": user_msgs,
                                "Assistant Messages": assistant_msgs,
                                "Total": user_msgs + assistant_msgs
                            })
                        
                        if conv_data:
                            st.dataframe(pd.DataFrame(conv_data))
                        else:
                            st.info("No conversation data available")
                    else:
                        st.info("No conversations available")
        
        # Tab 2.4: Entity Analysis
        with mem_tab4:
            st.header("Entity Analysis")
            if st.button("Analyze Entities"):
                with st.spinner("Analyzing entity relationships..."):
                    # Use get_entity_relationships (non-async method)
                    relationships = st.session_state.memory_llm.get_entity_relationships()
                    
                    if relationships:
                        for rel in relationships:
                            with st.expander(f"{rel['entity1']} ↔ {rel['entity2']}"):
                                st.write(f"Relationship Strength: {rel['relationship_strength']}")
                                st.write("Shared Contexts:")
                                for context in rel['shared_contexts']:
                                    st.write(f"- {context}")
                    else:
                        st.info("No significant entity relationships found")
            
            # Add manual memory maintenance
            st.subheader("Memory Maintenance")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Memory Decay"):
                    with st.spinner("Running memory decay..."):
                        await st.session_state.memory_llm._check_memory_decay()
                        st.success("Memory decay completed")
            with col2:
                if st.button("Run Memory Cleanup"):
                    with st.spinner("Running memory cleanup..."):
                        await st.session_state.memory_llm.cleanup_memories()
                        st.success("Memory cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())