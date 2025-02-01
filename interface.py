import streamlit as st
import asyncio
from episodic_memory import EpisodicMemoryModule, MemoryPriority
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

# Initialize session state
if 'memory_llm' not in st.session_state:
    st.session_state.memory_llm = EpisodicMemoryModule()
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = nx.Graph()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation' not in st.session_state:
    # Initialize Groq chat
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'llama3-8b-8192'
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )
    
    # Initialize conversation memory and chain
    system_prompt = 'You are a friendly conversational chatbot'
    conversational_memory_length = 5
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, 
        memory_key="chat_history", 
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    
    st.session_state.conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

async def update_knowledge_graph():
    """Update the knowledge graph with current entity relationships"""
    relationships = await st.session_state.memory_llm.analyze_entity_relationships()
    G = nx.Graph()
    
    for entity1, related in relationships.items():
        G.add_node(entity1)
        for rel in related:
            G.add_edge(
                entity1,
                rel['entity'],
                weight=rel['co_occurrences']
            )
    
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

async def main():
    st.title("AI Chat & Memory Dashboard")
    
    # Create tabs for Chat and Memory Management
    tab1, tab2 = st.tabs(["Chat", "Memory Management"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("user" if message.get('role') == 'user' else "assistant"):
                st.write(message.get('content'))
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.predict(human_input=prompt)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Store the interaction in episodic memory
            await st.session_state.memory_llm.store_memory(
                text=f"User: {prompt}\nAssistant: {response}",
                metadata={"category": "conversation"},
                priority=MemoryPriority.MEDIUM
            )
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
            
            col1, col2 = st.columns(2)
            with col1:
                days_ago = st.number_input("Days to Look Back", min_value=1, value=7)
            with col2:
                max_results = st.number_input("Max Results", min_value=1, value=5)
                
            if st.button("Search"):
                with st.spinner("Searching..."):
                    start_time = datetime.now() - timedelta(days=days_ago)
                    results = await st.session_state.memory_llm.recall_memories_advanced(
                        query=search_query,
                        filters={"category": search_category} if search_category else None,
                        time_range=(start_time, datetime.now()),
                        max_results=max_results
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
            st.plotly_chart(plot_knowledge_graph(), use_container_width=True)
        
        # Tab 2.3: Statistics
        with mem_tab3:
            st.header("Memory Statistics")
            stats = await st.session_state.memory_llm.get_memory_stats()
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
        
        # Tab 2.4: Entity Analysis
        with mem_tab4:
            st.header("Entity Analysis")
            if st.button("Analyze Entities"):
                with st.spinner("Analyzing entity relationships..."):
                    relationships = await st.session_state.memory_llm.get_entity_relationships()
                    if relationships:
                        for rel in relationships:
                            with st.expander(f"{rel['entity1']} ↔ {rel['entity2']}"):
                                st.write(f"Relationship Strength: {rel['relationship_strength']}")
                                st.write("Shared Contexts:")
                                for context in rel['shared_contexts']:
                                    st.write(f"- {context}")
                    else:
                        st.info("No significant entity relationships found")

if __name__ == "__main__":
    asyncio.run(main())