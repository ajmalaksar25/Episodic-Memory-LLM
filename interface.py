import streamlit as st
import asyncio
from episodic_memory import EpisodicMemoryLLM, MemoryPriority
from datetime import datetime, timedelta
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import json

# Initialize session state
if 'memory_llm' not in st.session_state:
    st.session_state.memory_llm = EpisodicMemoryLLM()
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = nx.Graph()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

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
    st.title("Episodic Memory LLM Dashboard")
    
    # Add auto-refresh functionality
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        st.sidebar.info(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Memory Input Section
    st.sidebar.header("Add New Memory")
    memory_text = st.sidebar.text_area("Memory Text")
    category = st.sidebar.text_input("Category")
    priority = st.sidebar.selectbox(
        "Priority",
        [MemoryPriority.LOW, MemoryPriority.MEDIUM, MemoryPriority.HIGH],
        format_func=lambda x: x.name
    )
    
    if st.sidebar.button("Store Memory"):
        if memory_text and category:
            with st.spinner("Storing memory..."):
                memory_id = await st.session_state.memory_llm.store_memory(
                    memory_text,
                    {"category": category},
                    priority
                )
                await update_knowledge_graph()
                st.sidebar.success(f"Memory stored with ID: {memory_id}")

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Memory Search", "Knowledge Graph", "Statistics", "Entity Analysis"
    ])
    
    # Tab 1: Memory Search
    with tab1:
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

if __name__ == "__main__":
    asyncio.run(main())