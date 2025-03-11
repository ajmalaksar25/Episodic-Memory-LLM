import streamlit as st
import asyncio
from .chatbot import MemoryBot
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'memory_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('MemoryBot')

# Initialize session state
if 'memory_bot' not in st.session_state:
    st.session_state.memory_bot = MemoryBot()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory_timeline' not in st.session_state:
    st.session_state.memory_timeline = []
if 'last_cleanup' not in st.session_state:
    st.session_state.last_cleanup = datetime.now()

def create_memory_network():
    """Create network visualization of memories and their relationships"""
    memories = st.session_state.memory_bot.memory_llm.collection.get()
    G = nx.Graph()
    
    # Add nodes for memories
    for idx, doc in enumerate(memories['documents']):
        G.add_node(f"mem_{idx}", type="memory", text=doc[:50] + "...")
        
    # Add edges based on similarity
    for i in range(len(memories['documents'])):
        for j in range(i + 1, len(memories['documents'])):
            similarity = 0.5  # Placeholder - implement actual similarity calculation
            if similarity > 0.3:
                G.add_edge(f"mem_{i}", f"mem_{j}", weight=similarity)
    
    # Create figure and draw network
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8, font_weight='bold')
    plt.close()  # Close the figure to free memory
    return fig

def plot_memory_decay():
    """Create visualization of memory importance decay over time"""
    memories = st.session_state.memory_timeline
    if not memories:
        return None
        
    df = pd.DataFrame(memories)
    fig = px.line(df, x='timestamp', y='importance', 
                  color='category',
                  title='Memory Importance Decay Over Time')
    return fig

def plot_memory_distribution():
    """Create visualization of memory distribution by category and priority"""
    stats = asyncio.run(st.session_state.memory_bot.get_memory_analysis())
    if not stats:
        return None
        
    df = pd.DataFrame(list(stats['categories'].items()), 
                     columns=['Category', 'Count'])
    fig = px.pie(df, values='Count', names='Category',
                 title='Memory Distribution by Category')
    return fig

def create_entity_graph():
    """Create visualization of entity relationships"""
    relationships = asyncio.run(
        st.session_state.memory_bot.memory_llm.analyze_entity_relationships()
    )
    G = nx.Graph()
    
    for entity1, related in relationships.items():
        G.add_node(entity1)
        for rel in related:
            G.add_edge(entity1, rel['entity'], 
                      weight=rel['co_occurrences'])
    
    # Create figure and draw network
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgreen',
            node_size=1000, font_size=8, font_weight='bold')
    plt.close()  # Close the figure to free memory
    return fig

async def process_message(message: str):
    """Process user message and update visualizations"""
    response = await st.session_state.memory_bot.process_message(message)
    
    # Update memory timeline
    memory = {
        'text': message,
        'timestamp': datetime.now(),
        'importance': 1.0,
        'category': 'user_input'
    }
    st.session_state.memory_timeline.append(memory)
    
    return response

def main():
    logger.info("Starting Memory Bot Interface")
    
    st.title("Memory Bot Interface")
    
    # Sidebar for system status and controls
    st.sidebar.header("System Status")
    st.sidebar.metric("Active Memories", 
                     len(st.session_state.memory_timeline))
    st.sidebar.metric("Time Since Last Cleanup",
                     f"{(datetime.now() - st.session_state.last_cleanup).seconds}s")
    
    if st.sidebar.button("Force Memory Cleanup"):
        asyncio.run(st.session_state.memory_bot.check_cleanup())
        st.session_state.last_cleanup = datetime.now()
    
    # Main chat interface
    st.header("Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response = asyncio.run(process_message(prompt))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Memory Visualizations
    st.header("Memory Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Memory Network", "Memory Decay", "Categories", "Entity Relations"
    ])
    
    with tab1:
        st.subheader("Memory Network")
        network_fig = create_memory_network()
        st.pyplot(network_fig)
    
    with tab2:
        st.subheader("Memory Decay Over Time")
        fig_decay = plot_memory_decay()
        if fig_decay:
            st.plotly_chart(fig_decay)
    
    with tab3:
        st.subheader("Memory Distribution")
        fig_dist = plot_memory_distribution()
        if fig_dist:
            st.plotly_chart(fig_dist)
    
    with tab4:
        st.subheader("Entity Relationships")
        entity_fig = create_entity_graph()
        st.pyplot(entity_fig)
    
    # Memory Details
    st.header("Memory Details")
    if st.session_state.memory_timeline:
        df_memories = pd.DataFrame(st.session_state.memory_timeline)
        st.dataframe(df_memories)

if __name__ == "__main__":
    main() 