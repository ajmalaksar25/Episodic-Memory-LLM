import re
import os
import sys
import uuid
import json
import asyncio
import chromadb
import traceback
from enum import Enum
from groq import Groq
from spacy import load
from threading import Lock
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from llm_providers import GroqProvider
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Union, Any
import math

# Load environment variables
load_dotenv()

class MemoryPriority(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class MemoryNode:
    id: str
    text: str
    timestamp: datetime
    category: str
    importance: float
    metadata: Dict
    references: int = 0
    last_accessed: datetime = None

@dataclass
class MemoryConfig:
    max_context_items: int = 10
    memory_decay_factor: float = 0.95
    importance_threshold: float = 0.5
    min_references_to_keep: int = 2
    
    # Add intervals for maintenance tasks
    decay_interval: int = 24  # hours
    cleanup_interval: int = 12  # hours
    decay_check_interval: int = 3600  # seconds
    
    # Default priority settings
    default_priority: float = 0.5
    priority_levels: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 0.3,
        "MEDIUM": 0.5,
        "HIGH": 0.8
    })

class EpisodicMemoryModule:
    def __init__(
        self,
        llm_provider: GroqProvider,
        collection_name: str = "episodic_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[MemoryConfig] = None,
        entity_config: Optional[Dict] = None,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        # Memory configuration
        self.config = config or MemoryConfig()
        
        # Entity extraction configuration
        self.entity_config = entity_config or {
            "PERSON": ["PERSON"],
            "ORGANIZATION": ["ORG"],
            "TECHNOLOGY": ["PRODUCT", "WORK_OF_ART", "SOFTWARE", "TOOL"],  # More specific
            "LOCATION": ["GPE", "LOC"],
            "CONCEPT": ["NORP", "EVENT", "LAW"],
            "MISC": ["LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"],
            "GENERAL": []  # New category for uncategorized terms
        }
        
        # Add technical term patterns
        self.tech_patterns = [
            r"(?i)(software|programming|code|api|framework|library|algorithm|database|server|cloud|interface|function|class|method|variable|data structure|protocol)",
            r"(?i)(python|java|javascript|c\+\+|ruby|golang|rust|sql|html|css|php)",
            r"(?i)(docker|kubernetes|aws|azure|git|linux|unix|windows|mac)",
        ]

        # Initialize vector store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Initialize or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

        # Neo4j initialization
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
        
        # Initialize Neo4j schema on startup
        self._init_neo4j_schema()
        
        # Other settings
        self.llm_provider = llm_provider
        self.context_window = []

        self._lock = Lock()

        # Add spaCy initialization
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load()

        # Start memory maintenance tasks
        self.maintenance_tasks = [
            asyncio.create_task(self._run_periodic_decay()),
            asyncio.create_task(self._run_periodic_cleanup())
        ]

    def _init_neo4j_schema(self):
        """Initialize Neo4j schema with consistent relationship types and constraints"""
        # Define schema constants
        self.RELATIONSHIP_TYPES = {
            "CONTAINS": "CONTAINS",           # Memory contains an entity
            "RELATED_TO": "RELATED_TO",       # Entity is related to another entity
            "FOLLOWS": "FOLLOWS",             # Memory follows another memory in sequence
            "PART_OF": "PART_OF",             # Memory is part of a conversation
            "REFERENCES": "REFERENCES"        # Memory references another memory
        }
        
        with self.driver.session() as session:
            # Create constraints for unique IDs
            try:
                session.run("CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE")
            except Exception as e:
                # For Neo4j versions that don't support the IF NOT EXISTS syntax
                print(f"Creating memory constraint with alternative syntax: {e}")
                try:
                    session.run("CREATE CONSTRAINT ON (m:Memory) ASSERT m.id IS UNIQUE")
                except:
                    print("Memory constraint may already exist, continuing...")
            
            try:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            except Exception as e:
                print(f"Creating entity constraint with alternative syntax: {e}")
                try:
                    session.run("CREATE CONSTRAINT ON (e:Entity) ASSERT e.id IS UNIQUE")
                except:
                    print("Entity constraint may already exist, continuing...")
            
            # Create indexes for faster queries
            try:
                session.run("CREATE INDEX memory_importance IF NOT EXISTS FOR (m:Memory) ON (m.importance)")
            except Exception as e:
                print(f"Creating index with alternative syntax: {e}")
                try:
                    session.run("CREATE INDEX ON :Memory(importance)")
                except:
                    print("Memory importance index may already exist, continuing...")
            
            try:
                session.run("CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)")
            except:
                try:
                    session.run("CREATE INDEX ON :Memory(timestamp)")
                except:
                    print("Memory timestamp index may already exist, continuing...")
            
            try:
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            except:
                try:
                    session.run("CREATE INDEX ON :Entity(type)")
                except:
                    print("Entity type index may already exist, continuing...")
            
            print("Neo4j schema initialized with constraints and indexes")

    def _create_memory_with_entities(self, tx, memory_id: str, text: str, metadata: Dict, entities: Dict):
        """Create a memory node with consistent relationship types and entity connections"""
        # Create memory node
        memory_query = """
        CREATE (m:Memory {
        id: $id,
            text: $text,
        timestamp: $timestamp,
            importance: $importance,
        category: $category,
            conversation_id: $conversation_id,
        references: 0,
        last_accessed: $timestamp
        })
        RETURN m
        """
            
        # Extract metadata with defaults
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        importance = metadata.get("importance", 0.5)
        category = metadata.get("category", "general")
        conversation_id = metadata.get("conversation_id", "default")
        
        # Create memory node
        tx.run(
            memory_query,
            id=memory_id,
            text=text,
            timestamp=timestamp,
            importance=importance,
            category=category,
            conversation_id=conversation_id
        )
        
        # Process entities
        for entity_type, entity_values in entities.items():
            for entity_value in entity_values:
                # Create unique ID for entity
                entity_id = f"{entity_type.lower()}_{re.sub(r'[^a-z0-9]', '_', entity_value.lower())}"
                
                # Create or merge entity node
                entity_query = """
                MERGE (e:Entity {id: $entity_id})
                ON CREATE SET e.value = $value, e.type = $type, e.created_at = $timestamp
                ON MATCH SET e.last_referenced = $timestamp
                        WITH e
                
                        MATCH (m:Memory {id: $memory_id})
                        MERGE (m)-[r:CONTAINS]->(e)
                RETURN e
                """
                
                tx.run(
                    entity_query,
                    entity_id=entity_id,
                    value=entity_value,
                    type=entity_type,
                    timestamp=timestamp,
                    memory_id=memory_id
                )
        
        # If conversation_id is provided, link memory to conversation
        if conversation_id and conversation_id != "default":
            conversation_query = """
            MERGE (c:Conversation {id: $conversation_id})
            ON CREATE SET c.created_at = $timestamp
            WITH c
            
            MATCH (m:Memory {id: $memory_id})
            MERGE (m)-[r:PART_OF]->(c)
            RETURN c
            """
            
            tx.run(
                conversation_query,
                conversation_id=conversation_id,
                timestamp=timestamp,
                memory_id=memory_id
            )

    async def store_memory(
        self,
        text: str,
        metadata: Dict = None,
        conversation_id: str = None,
        priority: Union[MemoryPriority, str] = MemoryPriority.MEDIUM,
        skip_entity_extraction: bool = False
    ) -> str:
        """
        Store a new memory with improved importance calculation and entity extraction.
        
        Args:
            text: The text to store as a memory
            metadata: Optional metadata to associate with the memory
            conversation_id: Optional conversation ID to associate with the memory
            priority: Priority level for the memory (affects importance)
            skip_entity_extraction: Whether to skip entity extraction
            
        Returns:
            ID of the stored memory
        """
        if not text:
            return None
            
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add conversation ID to metadata if provided
        if conversation_id:
            metadata["conversation_id"] = conversation_id
            
        # Add priority to metadata
        if isinstance(priority, MemoryPriority):
            metadata["priority"] = priority.value
        else:
            metadata["priority"] = priority
            
        # Calculate importance score
        importance = await self._calculate_memory_importance(text, metadata)
        
        # Current timestamp
        timestamp = datetime.now().isoformat()
        
        # Add memory to vector store
        self.collection.add(
            documents=[text],
            metadatas=[{
                "id": memory_id,
                "timestamp": timestamp,
                "importance": importance,
                **metadata
            }],
            ids=[memory_id]
        )
        
        # Extract entities if not skipped
        entities = {}
        if not skip_entity_extraction:
            entities = await self._extract_entities(text)
        
        # Store memory in Neo4j with entities
        with self.driver.session() as session:
            session.execute_write(
                self._create_memory_with_entities,
                memory_id=memory_id,
                text=text,
                metadata={
                    "timestamp": timestamp,
                    "importance": importance,
                    **metadata
                },
                entities=entities
            )
            
        # Update context window with the new memory
        self._update_context_window([{
            "id": memory_id,
            "text": text,
            "timestamp": timestamp,
            "importance": importance,
            "metadata": metadata,
            "entities": entities
        }])
            
        return memory_id

    async def recall_memories(
        self, 
        query: str, 
        category: str = None, 
        conversation_id: str = None, 
        min_importance: float = 0.0, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Recall memories related to a query, with optional filtering by category and conversation.
        
        Args:
            query: The query to search for related memories
            category: Optional category to filter memories
            conversation_id: Optional conversation ID to filter memories
            min_importance: Minimum importance score for memories to be returned
            top_k: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries with their text and metadata
        """
        # Check cache first for identical queries to improve performance
        cache_key = f"{query}_{category}_{conversation_id}_{min_importance}_{top_k}"
        if hasattr(self, '_memory_cache') and cache_key in self._memory_cache:
            # Update access time for cached memories
            for memory_id in [m['id'] for m in self._memory_cache[cache_key]]:
                with self.driver.session() as session:
                    session.execute_write(self._update_memory_access_tx, memory_id)
            return self._memory_cache[cache_key]
        
        # Build where clause for filtering
        where_clause = self._build_where_clause(category, conversation_id, min_importance)
            
        # Get related memories using vector similarity and graph relationships
        with self.driver.session() as session:
            memories = session.execute_read(self._get_related_memories_tx, query, where_clause, top_k)
            
        # Update access time for retrieved memories
        for memory_id in [m['id'] for m in memories]:
            with self.driver.session() as session:
                session.execute_write(self._update_memory_access_tx, memory_id)
        
        # Update context window with new memories
        self._update_context_window(memories)
        
        # Cache the results for future identical queries
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
        self._memory_cache[cache_key] = memories
        
        # Limit cache size to prevent memory leaks
        if len(self._memory_cache) > 100:  # Arbitrary limit, adjust as needed
            # Remove oldest cache entries
            oldest_keys = sorted(self._memory_cache.keys())[:50]
            for key in oldest_keys:
                del self._memory_cache[key]
        
        return memories

    def _get_memory_context(self, tx, memory_id: str) -> Dict:
        """Get related entities and contexts for a memory from Neo4j"""
        if not memory_id:
            return {"entities": [], "contexts": []}
        
        query = """
        MATCH (m:Memory {id: $memory_id})-[r:CONTAINS]->(e:Entity)
        WHERE m IS NOT NULL AND e IS NOT NULL
        RETURN e.name as entity_name,
               r.context_count as context_count,
               r.timestamp as relationship_timestamp
        """
        result = tx.run(query, memory_id=memory_id)
        records = result.data()
        
        return {
            "entities": [record["entity_name"] for record in records],
            "contexts": [{"count": record["context_count"], 
                         "timestamp": record["relationship_timestamp"]} 
                       for record in records]
        }

    def _update_memory_access_tx(self, tx, memory_id: str):
        query = """
        MATCH (m:Memory {id: $memory_id})
        SET m.last_accessed = datetime(),
            m.references = coalesce(m.references, 0) + 1
        """
        tx.run(query, memory_id=memory_id)
        
    def get_entity_relationships(self) -> List[Dict]:
        """Get relationships between entities with improved error handling"""
        with self.driver.session() as session:
            try:
                query = """
                MATCH (e1:Entity)<-[r1:CONTAINS]-(m:Memory)-[r2:CONTAINS]->(e2:Entity)
                WHERE e1.name < e2.name  // Avoid duplicate relationships
                WITH e1, e2, collect(DISTINCT m) as shared_memories,
                     count(DISTINCT m) as shared_count
                WHERE shared_count > 0
                RETURN {
                    entity1: e1.name,
                    entity2: e2.name,
                    entity1_type: e1.type,
                    entity2_type: e2.type,
                    shared_memory_count: shared_count,
                    relationship_strength: toFloat(shared_count),
                    shared_contexts: [mem in shared_memories | mem.text][..5]  // Limit to 5 examples
                } as relationship
                ORDER BY shared_count DESC
                LIMIT 10
                """
                
                result = session.run(query)
                relationships = [record["relationship"] for record in result]
                print(f"Debug: Found {len(relationships)} entity relationships")
                
                if relationships:
                    print("Entity names:", set([r["entity1"] for r in relationships] + 
                                             [r["entity2"] for r in relationships]))
                
                return relationships
                    
            except Exception as e:
                print(f"Error getting entity relationships: {e}")
                print(f"Full stacktrace:", traceback.format_exc())
                return []
        
    def _build_where_clause(
        self, 
        category: Optional[str], 
        conversation_id: Optional[str], 
        min_importance: Union[float, int]
    ) -> Dict:
        """Build where clause with type checking"""
        if not isinstance(min_importance, (float, int)):
            min_importance = float(min_importance)
        if category and not conversation_id:
            return {"category": category}
        elif conversation_id and not category:
            return {"conversation_id": conversation_id}
        elif category and conversation_id:
            return {
                "$and": [
                    {"category": category},
                    {"conversation_id": conversation_id},
                    {"importance": {"$gte": min_importance}} if min_importance > 0 else {"importance": {"$gte": 0}}
                ]
            }
        elif min_importance > 0:
            return {"importance": {"$gte": min_importance}}
        else:
            return {}
        
    def _get_related_memories_tx(self, tx, query: str, where_clause: Dict, top_k: int = 5) -> List[Dict]:
        """
        Get memories related to a query with optimized performance for large memory stores.
        
        Args:
            tx: Neo4j transaction
            query: The search query
            where_clause: Filtering conditions
            top_k: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        # Get vector embedding for the query
        query_embedding = self.embedding_function([query])[0]
        
        # Skip full-text search as it requires APOC or Neo4j Enterprise
        # Use vector similarity search
        # Get embeddings from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, 20),  # Get more results than needed to filter
            where=where_clause
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            return []
        
        # Extract memory IDs and distances
        memory_ids = results['ids'][0]
        distances = results['distances'][0] if 'distances' in results else [1.0] * len(memory_ids)
        
        # Convert distances to relevance scores (1 - distance for cosine distance)
        relevance_scores = [1 - dist for dist in distances]
        
        # Prepare parameters for Neo4j query
        memory_id_relevance = {
            memory_id: relevance 
            for memory_id, relevance in zip(memory_ids, relevance_scores)
        }
        
        # Batch process to avoid large parameter lists
        batch_size = 10
        all_memories = []
        
        for i in range(0, len(memory_ids), batch_size):
            batch_ids = memory_ids[i:i+batch_size]
            
            # Get memory details from Neo4j
            query = """
        MATCH (m:Memory)
            WHERE m.id IN $memory_ids
            """
            
            # Add where clause conditions
            if where_clause.get("category"):
                query += " AND m.category = $category"
            if where_clause.get("conversation_id"):
                query += " AND m.conversation_id = $conversation_id"
            if where_clause.get("min_importance"):
                query += " AND m.importance >= $min_importance"
            
            query += """
            RETURN m.id AS id, m.text AS text, m.importance AS importance, 
                   m.timestamp AS timestamp, m.category AS category
        """
        
        params = {
                "memory_ids": batch_ids,
                **where_clause
            }
            
        result = tx.run(query, params)
        batch_memories = [dict(record) for record in result]
        
        # Add relevance scores
        for memory in batch_memories:
            memory["relevance_score"] = memory_id_relevance.get(memory["id"], 0)
            memory["relevance"] = round(memory["relevance_score"], 3)
        
        all_memories.extend(batch_memories)
        
        # Sort by relevance and limit to top_k
        all_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_memories[:top_k]

    async def _check_memory_decay(self):
        """
        Check and apply memory decay using an enhanced forgetting curve model.
        
        This implements a more sophisticated memory decay mechanism based on:
        1. Ebbinghaus forgetting curve
        2. Spaced repetition principles
        3. Importance-based retention
        """
        try:
            # Get current time
            current_time = datetime.now()
            
            # Calculate decay threshold (memories older than this will be considered for decay)
            decay_threshold = (current_time - timedelta(hours=self.config.decay_interval)).isoformat()
            
            # Apply memory decay
            with self.driver.session() as session:
                session.execute_write(
                    self._apply_enhanced_memory_decay,
                    decay_threshold=decay_threshold,
                    importance_threshold=self.config.importance_threshold,
                    min_references=self.config.min_references_to_keep
                )
                
        except Exception as e:
            print(f"Error checking memory decay: {e}")
    
    def _apply_enhanced_memory_decay(self, tx, decay_threshold: str, importance_threshold: float, min_references: int):
        """
        Apply enhanced memory decay using a sophisticated forgetting curve model.
        
        Args:
            tx: Neo4j transaction
            decay_threshold: Timestamp threshold for decay consideration
            importance_threshold: Minimum importance to keep memories
            min_references: Minimum references to keep memories regardless of importance
        """
        # Query to get memories eligible for decay calculation
        query = """
        MATCH (m:Memory)
        WHERE m.timestamp < $decay_threshold
        RETURN m.id AS id, m.importance AS importance, m.references AS references,
               m.timestamp AS created_at, m.last_accessed AS last_accessed
        """
        
        result = tx.run(query, decay_threshold=decay_threshold)
        memories_to_update = []
        memories_to_remove = []
        
        for record in result:
            memory_id = record["id"]
            importance = record["importance"]
            references = record["references"] or 0
            created_at = datetime.fromisoformat(record["created_at"].replace('Z', '+00:00') if record["created_at"].endswith('Z') else record["created_at"])
            
            # Use last_accessed if available, otherwise use created_at
            last_accessed = None
            if record["last_accessed"]:
                try:
                    last_accessed = datetime.fromisoformat(record["last_accessed"].replace('Z', '+00:00') if record["last_accessed"].endswith('Z') else record["last_accessed"])
                except:
                    last_accessed = created_at
            else:
                last_accessed = created_at
            
            # Calculate time factors
            time_since_creation = (datetime.now() - created_at).total_seconds() / 86400  # days
            time_since_access = (datetime.now() - last_accessed).total_seconds() / 86400  # days
            
            # Calculate decay factors
            
            # 1. Base decay from Ebbinghaus forgetting curve: R = e^(-t/S)
            # where R is retention, t is time, S is strength (influenced by importance)
            strength = 2.0 + (importance * 8.0)  # Scale importance to strength factor (2-10)
            base_decay = math.exp(-time_since_creation / strength)
            
            # 2. Access recency factor: more recent access = less decay
            recency_factor = math.exp(-time_since_access / 5.0)  # 5-day half-life for access recency
            
            # 3. References factor: more references = less decay
            reference_factor = min(1.0, references / 5.0)  # Max benefit at 5 references
            
            # Combine factors with weights
            # 50% base decay, 30% recency, 20% references
            combined_decay = (0.5 * base_decay) + (0.3 * recency_factor) + (0.2 * reference_factor)
            
            # Calculate new importance
            new_importance = importance * combined_decay
            
            # Decide whether to update or remove the memory
            if new_importance < importance_threshold and references < min_references:
                memories_to_remove.append(memory_id)
            else:
                memories_to_update.append({
                    "id": memory_id,
                    "importance": new_importance
                })
        
        # Update memories in batches
        batch_size = 50
        for i in range(0, len(memories_to_update), batch_size):
            batch = memories_to_update[i:i+batch_size]
            
            # Skip if batch is empty
            if not batch:
                continue
                
            # Prepare parameters for batch update
            params = {
                f"id_{j}": mem["id"] for j, mem in enumerate(batch)
            }
            params.update({
                f"importance_{j}": mem["importance"] for j, mem in enumerate(batch)
            })
            
            # Build query for batch update
            update_query = "UNWIND ["
            for j in range(len(batch)):
                if j > 0:
                    update_query += ", "
                update_query += f"{{id: $id_{j}, importance: $importance_{j}}}"
            update_query += """] AS data
            MATCH (m:Memory {id: data.id})
            SET m.importance = data.importance
            """
            
            tx.run(update_query, **params)
        
        # Remove memories in batches
        for i in range(0, len(memories_to_remove), batch_size):
            batch = memories_to_remove[i:i+batch_size]
            
            # Skip if batch is empty
            if not batch:
                continue
                
            # Prepare parameters for batch removal
            params = {
                "ids": batch
            }
            
            # Remove from Neo4j
            remove_query = """
        MATCH (m:Memory)
            WHERE m.id IN $ids
            DETACH DELETE m
            """
            
            tx.run(remove_query, **params)
            
            # Also remove from vector store
            try:
                self.collection.delete(ids=batch)
            except Exception as e:
                print(f"Error removing memories from vector store: {e}")
        
        print(f"Memory decay applied: {len(memories_to_update)} memories updated, {len(memories_to_remove)} memories removed")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory store"""
        stats = {
            "total_memories": 0,
            "total_entities": 0,
            "avg_importance": 0,
            "categories": {},
            "entity_types": {}
        }
        
        try:
            with self.driver.session() as session:
                # Get memory statistics
                memory_query = """
                MATCH (m:Memory)
                RETURN count(m) AS total_memories,
                       avg(m.importance) AS avg_importance
                """
                
                memory_result = session.run(memory_query)
                if memory_result.peek():
                    record = memory_result.single()
                    stats["total_memories"] = record["total_memories"]
                    stats["avg_importance"] = record["avg_importance"] or 0
                
                # Get entity statistics
                entity_query = """
                MATCH (e:Entity)
                RETURN count(e) AS total_entities
                """
                
                entity_result = session.run(entity_query)
                if entity_result.peek():
                    stats["total_entities"] = entity_result.single()["total_entities"]
                
                # Get category distribution
                category_query = """
                MATCH (m:Memory)
                WITH m.category AS category, count(*) AS count
                WHERE category IS NOT NULL
                RETURN category, count
                ORDER BY count DESC
                LIMIT 5
                """
                
                category_result = session.run(category_query)
                for record in category_result:
                    stats["categories"][record["category"]] = record["count"]
                
                # Get entity type distribution
                entity_type_query = """
                MATCH (e:Entity)
                WITH e.type AS type, count(*) AS count
                WHERE type IS NOT NULL
                RETURN type, count
                ORDER BY count DESC
                LIMIT 5
                """
                
                entity_type_result = session.run(entity_type_query)
                for record in entity_type_result:
                    stats["entity_types"][record["type"]] = record["count"]
                
            return stats
                
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}

    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using a combination of NLP techniques with robust JSON parsing.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of entity types to lists of entity values
        """
        try:
            # Use spaCy for initial entity extraction
            doc = self.nlp(text)
            
            # Extract named entities from spaCy
            entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],  # Geopolitical entities (countries, cities)
                "LOC": [],  # Non-GPE locations
                "PRODUCT": [],
                "EVENT": [],
                "DATE": [],
                "TIME": [],
                "CONCEPT": [],  # Custom category for abstract concepts
                "TOPIC": []     # Custom category for discussion topics
            }
            
            # Extract entities from spaCy
            for ent in doc.ents:
                if ent.label_ in entities:
                    # Clean and normalize the entity text
                    clean_text = ent.text.strip().lower()
                    if clean_text and clean_text not in entities[ent.label_]:
                        entities[ent.label_].append(clean_text)
            
            # Use LLM to extract additional entities, especially concepts and topics
            # that might be missed by spaCy
            if self.llm_provider:
                # Define a list of prompt templates with increasing structure
                prompt_templates = [
                    # Template 1: Simple JSON request
                    """
                    Extract key entities from the following text. Focus on:
                    1. CONCEPTS: Abstract ideas or principles
                    2. TOPICS: Main subjects of discussion
                    3. Any important entities missed by standard NLP
                    
                    Text: "{text}"
                    
                    Format your response as a JSON object with these categories.
                    Example format:
                    {{
                      "CONCEPTS": ["artificial intelligence", "machine learning"],
                      "TOPICS": ["data science", "neural networks"],
                      "PERSON": ["Alan Turing"]
                    }}
                    """,
                    
                    # Template 2: More structured with explicit instructions
                    """
                    Your task is to extract entities from the text below and return ONLY a valid JSON object.
                    
                    Text: "{text}"
                    
                    Return a JSON object with these exact keys (include only if entities are found):
                    - PERSON: list of people mentioned
                    - ORG: list of organizations mentioned
                    - GPE: list of countries, cities, states mentioned
                    - LOC: list of locations mentioned
                    - PRODUCT: list of products mentioned
                    - EVENT: list of events mentioned
                    - DATE: list of dates mentioned
                    - TIME: list of times mentioned
                    - CONCEPT: list of abstract concepts mentioned
                    - TOPIC: list of main topics discussed
                    
                    Example of valid response:
                    {{
                      "PERSON": ["John Smith"],
                      "CONCEPT": ["artificial intelligence"],
                      "TOPIC": ["machine learning"]
                    }}
                    
                    IMPORTANT: Return ONLY the JSON object, nothing else.
                    """,
                    
                    # Template 3: Extremely structured with field-by-field extraction
                    """
                    Extract entities from this text: "{text}"
                    
                    For each category below, list entities as comma-separated values, or write "none" if none found:
                    
                    PERSON: 
                    ORG: 
                    GPE: 
                    LOC: 
                    PRODUCT: 
                    EVENT: 
                    DATE: 
                    TIME: 
                    CONCEPT: 
                    TOPIC: 
                    """
                ]
                
                # Try each prompt template until successful
                llm_entities = {}
                for i, template in enumerate(prompt_templates):
                    try:
                        prompt = template.format(text=text)
                        llm_response = await self.llm_provider.generate_text(prompt, max_tokens=400)
                        
                        # For the third template, parse the structured format
                        if i == 2:
                            # Parse line by line format
                            parsed_entities = {}
                            lines = llm_response.strip().split('\n')
                            for line in lines:
                                if ':' in line:
                                    category, values = line.split(':', 1)
                                    category = category.strip().upper()
                                    if category in entities and values.strip().lower() != "none":
                                        parsed_entities[category] = [v.strip().lower() for v in values.strip().split(',')]
                            
                            if parsed_entities:
                                llm_entities = parsed_entities
                        break
                    except Exception as e:
                        print(f"Attempt {i+1} failed: {e}")
                        # Continue to the next template if this one failed
                
                # Merge LLM-extracted entities with spaCy entities
                for entity_type, values in llm_entities.items():
                    if isinstance(values, list):
                        entity_type_upper = entity_type.upper()
                        if entity_type_upper in entities:
                            for value in values:
                                if isinstance(value, str):
                                    clean_value = value.strip().lower()
                                    if clean_value and clean_value not in entities[entity_type_upper]:
                                        entities[entity_type_upper].append(clean_value)
            
            # Remove empty categories and return
            return {k: v for k, v in entities.items() if v}
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return {}

    def _update_context_window(self, new_memories: List[Dict]):
        """Update context window with efficient memory management and error handling"""
        try:
            if not isinstance(new_memories, list):
                raise ValueError("new_memories must be a list")
            
            current_time = datetime.now()
            
            try:
                # Update importance based on decay and time since last access
                for memory in self.context_window:
                    if "last_accessed" in memory:
                        time_diff = (current_time - memory["last_accessed"]).total_seconds() / 3600
                        decay = max(0.5, self.config.memory_decay_factor ** (time_diff / 24))
                        memory["importance"] = memory.get("importance", 0.5) * decay
                
                # Remove duplicates and low importance memories
                seen_texts = set()
                filtered_window = []
                
                for mem in self.context_window:
                    text_key = mem.get("text", "").strip().lower()
                    if (mem.get("importance", 0) > self.config.importance_threshold and 
                        text_key not in seen_texts):
                        seen_texts.add(text_key)
                        filtered_window.append(mem)
                
                # Add new memories
                for memory in new_memories:
                    if not isinstance(memory, dict):
                        print(f"Skipping invalid memory format: {memory}")
                        continue
                    
                    text_key = memory.get("text", "").strip().lower()
                    if text_key and text_key not in seen_texts:
                        seen_texts.add(text_key)
                        memory_copy = memory.copy()
                        memory_copy["last_accessed"] = current_time
                        filtered_window.append(memory_copy)
                
                # Sort and limit
                self.context_window = sorted(
                    filtered_window,
                    key=lambda x: (
                        x.get("importance", 0) * 0.7 + 
                        (1 - (current_time - x.get("last_accessed", current_time)).total_seconds() / 86400) * 0.3
                    ),
                    reverse=True
                )[:self.config.max_context_items]
                
            except Exception as e:
                print(f"Error processing context window: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error updating context window: {e}")
            traceback.print_exc()

    def get_context_window(self) -> List[Dict]:
        """Get current context window contents with error handling"""
        try:
            return [
                {
                    "text": mem.get("text", ""),
                    "importance": mem.get("importance", 0),
                    "last_accessed": mem.get("last_accessed", datetime.now()).isoformat(),
                    "category": mem.get("category", "general")
                }
                for mem in self.context_window
                if isinstance(mem, dict) and "text" in mem
            ]
        except Exception as e:
            print(f"Error retrieving context window: {e}")
            traceback.print_exc()
            return []

    async def cleanup_memories(self):
        """
        Clean up the memory store by removing duplicates and low-importance memories.
        
        Returns:
            Tuple of (duplicates_removed, orphaned_entities_removed)
        """
        duplicates_removed = 0
        orphaned_entities_removed = 0
        
        try:
            # 1. Find and remove duplicate memories
            with self.driver.session() as session:
                # Find potential duplicates using text similarity
                duplicates = session.execute_read(self._find_duplicate_memories_tx)
                
                # Process each set of duplicates
                for duplicate_set in duplicates:
                    # Keep the memory with highest importance or most recent if tied
                    memories_to_keep = duplicate_set[0]
                    memories_to_remove = duplicate_set[1:]
                    
                    # Remove the duplicates
                    for memory_id in memories_to_remove:
                        session.execute_write(self._remove_memory_tx, memory_id)
                        duplicates_removed += 1
                        
                        # Also remove from vector store
                        try:
                            self.collection.delete(ids=[memory_id])
                        except Exception as e:
                            print(f"Error removing from vector store: {e}")
            
            # 2. Find and remove orphaned entities (entities with no connected memories)
            with self.driver.session() as session:
                orphaned = session.execute_read(self._find_orphaned_entities_tx)
                
                # Remove orphaned entities
                for entity_id in orphaned:
                    session.execute_write(self._remove_entity_tx, entity_id)
                    orphaned_entities_removed += 1
            
            print(f"Cleaned up {duplicates_removed} duplicate memories")
            print(f"Cleaned up {orphaned_entities_removed} orphaned entities")
            
            return duplicates_removed, orphaned_entities_removed
            
        except Exception as e:
            print(f"Error during memory cleanup: {e}")
            return 0, 0
    
    def _find_duplicate_memories_tx(self, tx):
        """Find duplicate memories based on text similarity with APOC fallback"""
        # Try using APOC for better similarity detection
        apoc_query = """
        MATCH (m1:Memory)
        MATCH (m2:Memory)
        WHERE m1.id < m2.id  // Avoid comparing the same pair twice
        AND m1.text IS NOT NULL AND m2.text IS NOT NULL
        // Calculate Jaccard similarity between memory texts
        WITH m1, m2, apoc.text.jaroWinklerDistance(m1.text, m2.text) AS similarity
        WHERE similarity > 0.9  // High similarity threshold
        // Group by first memory and collect potential duplicates
        RETURN m1.id AS original, collect(m2.id) AS duplicates
        """
        
        try:
            # Try APOC first
            result = tx.run(apoc_query)
            duplicate_sets = []
            
            for record in result:
                original = record["original"]
                duplicates = record["duplicates"]
                if duplicates:
                    duplicate_sets.append([original] + duplicates)
                    
            return duplicate_sets
            
        except Exception as e:
            print(f"APOC similarity function not available, falling back to exact matching: {e}")
            
            # Fallback to exact matching if APOC is not available
            fallback_query = """
                    MATCH (m1:Memory)
                    MATCH (m2:Memory)
                    WHERE m1.id < m2.id 
                    AND m1.text = m2.text
                    AND m1.conversation_id = m2.conversation_id
            RETURN m1.id AS original, collect(m2.id) AS duplicates
            """
            
            result = tx.run(fallback_query)
            duplicate_sets = []
            
            for record in result:
                original = record["original"]
                duplicates = record["duplicates"]
                if duplicates:
                    duplicate_sets.append([original] + duplicates)
                    
            return duplicate_sets
    
    def _remove_memory_tx(self, tx, memory_id):
        """Remove a memory and its relationships"""
        query = """
        MATCH (m:Memory {id: $memory_id})
        OPTIONAL MATCH (m)-[r]-()
        DELETE r, m
        """
        
        tx.run(query, memory_id=memory_id)
    
    def _find_orphaned_entities_tx(self, tx):
        """Find entities with no connected memories"""
        query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[:CONTAINS]->(:Memory)
        RETURN e.id AS entity_id
        """
        
        result = tx.run(query)
        return [record["entity_id"] for record in result]
    
    def _remove_entity_tx(self, tx, entity_id):
        """Remove an entity and its relationships"""
        query = """
        MATCH (e:Entity {id: $entity_id})
        OPTIONAL MATCH (e)-[r]-()
        DELETE r, e
        """
        
        tx.run(query, entity_id=entity_id)

    # Add destructor for cleanup
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Cancel maintenance tasks
            for task in self.maintenance_tasks:
                task.cancel()
            
            # Close Neo4j connection
            if hasattr(self, 'driver'):
                self.driver.close()
        except Exception as e:
            print(f"Error in cleanup: {e}")

    def validate_metadata(self, metadata: Dict) -> Dict:
        """Validate metadata before storage"""
        required_fields = [
            "category", 
            "importance", 
            "sentiment", 
            "priority",
            "conversation_id",  # Add conversation_id
            "timestamp"
        ]
        for field in required_fields:
            if field not in metadata:
                metadata[field] = self._get_default_value(field)
        return metadata

    def _get_default_value(self, field: str) -> Any:
        """Get default values for metadata fields"""
        defaults = {
            "category": "general",
            "importance": 0.5,
            "sentiment": "neutral",
            "priority": MemoryPriority.MEDIUM.value,
            "conversation_id": "default",  # Add default conversation_id
            "timestamp": datetime.now().isoformat()
        }
        return defaults.get(field)

    async def _run_periodic_decay(self):
        """Run periodic memory decay checks"""
        while True:
            try:
                await self._check_memory_decay()
                await asyncio.sleep(self.config.decay_check_interval)
            except Exception as e:
                print(f"Error in decay check: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _run_periodic_cleanup(self):
        """Run periodic memory cleanup"""
        while True:
            try:
                await self.cleanup_memories()
                await asyncio.sleep(self.config.cleanup_interval)
            except Exception as e:
                print(f"Error in cleanup: {e}")
                await asyncio.sleep(60)
                
    async def get_conversations(self) -> List[str]:
        """Get list of unique conversation IDs"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)
                WHERE m.conversation_id IS NOT NULL
                RETURN DISTINCT m.conversation_id as conversation_id
                ORDER BY m.conversation_id
            """)
            return [record["conversation_id"] for record in result]

    async def validate_conversation(self, conversation_id: str) -> bool:
        """Check if a conversation exists"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory {conversation_id: $conversation_id})
                RETURN count(m) > 0 as exists
            """, conversation_id=conversation_id)
            return result.single()["exists"]

    async def _calculate_memory_importance(self, text: str, metadata: Dict = None) -> float:
        """
        Calculate the importance of a memory based on its content and metadata.
        
        Args:
            text: The memory text
            metadata: Optional metadata associated with the memory
            
        Returns:
            Importance score between 0 and 1
        """
        # Start with base importance from metadata or default
        base_importance = 0.5
        if metadata and 'priority' in metadata:
            priority = metadata['priority']
            if isinstance(priority, str) and priority in self.config.priority_levels:
                base_importance = self.config.priority_levels[priority]
            elif isinstance(priority, (int, float)) and 0 <= priority <= 1:
                base_importance = priority
        
        # Use LLM to assess importance if available
        if self.llm_provider:
            try:
                prompt = f"""
                On a scale of 0.0 to 1.0, rate the importance of the following information for long-term memory:
                
                "{text}"
                
                Consider these factors:
                1. Uniqueness of information
                2. Potential future relevance
                3. Emotional significance
                4. Factual density
                5. Specificity (specific details vs general statements)
                
                Return only a single number between 0.0 and 1.0.
                """
                
                response = await self.llm_provider.generate_text(prompt, max_tokens=10)
                # Extract the numeric value from the response
                match = re.search(r'(\d+\.\d+|\d+)', response)
                if match:
                    llm_importance = float(match.group(0))
                    # Ensure the value is between 0 and 1
                    llm_importance = max(0.0, min(1.0, llm_importance))
                    
                    # Combine base importance with LLM assessment
                    # Weight LLM assessment more heavily (70%)
                    importance = (0.3 * base_importance) + (0.7 * llm_importance)
                    return importance
            except Exception as e:
                print(f"Error calculating memory importance with LLM: {e}")
        
        # If LLM assessment failed or not available, use heuristics
        
        # 1. Length factor - longer text might contain more information
        length_factor = min(len(text) / 500, 1.0)  # Normalize to max of 1.0
        
        # 2. Entity density - more entities might indicate more important information
        entities = {}
        try:
            doc = self.nlp(text)
            entities = {ent.label_: ent.text for ent in doc.ents}
        except:
            pass
        entity_factor = min(len(entities) / 5, 1.0)  # Normalize to max of 1.0
        
        # 3. Question factor - text containing questions might be more important
        question_factor = 0.0
        if '?' in text:
            question_factor = 0.2
        
        # 4. Recency factor - more recent memories might be more important
        recency_factor = 0.1  # Default small boost for new memories
        
        # Combine factors with base importance
        importance = base_importance + (0.2 * length_factor) + (0.2 * entity_factor) + question_factor + recency_factor
        
        # Ensure the final importance is between 0 and 1
        return max(0.0, min(1.0, importance))

    def _get_conversation_memories_tx(self, tx, conversation_id: str) -> List[Dict]:
        """
        Get memories from a specific conversation.
        
        Args:
            tx: Neo4j transaction
            conversation_id: ID of the conversation to retrieve memories from
            
        Returns:
            List of memory dictionaries
        """
        query = """
        MATCH (m:Memory)
        WHERE m.conversation_id = $conversation_id
        RETURN m.id as id,
               m.text as text,
               m.importance as importance,
               m.category as category,
               m.timestamp as timestamp
        ORDER BY m.timestamp DESC
        """
        result = tx.run(query, conversation_id=conversation_id)
        return [dict(record) for record in result]

    async def batch_store_memories(self, texts: List[str], metadata_list: List[Dict] = None, conversation_id: str = None) -> List[str]:
        """
        Store multiple memories in parallel for improved performance.
        
        Args:
            texts: List of text strings to store as memories
            metadata_list: Optional list of metadata dictionaries (one per text)
            conversation_id: Optional conversation ID to associate with all memories
            
        Returns:
            List of memory IDs for the stored memories
        """
        if not texts:
            return []
            
        # Ensure metadata_list is the same length as texts
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list must be the same length as texts")
            
        # Process memories in parallel
        import asyncio
        
        # Create tasks for calculating importance and extracting entities
        importance_tasks = []
        entity_tasks = []
        
        for i, text in enumerate(texts):
            metadata = metadata_list[i] or {}
            if conversation_id and "conversation_id" not in metadata:
                metadata["conversation_id"] = conversation_id
                
            # Create task for importance calculation
            importance_tasks.append(self._calculate_memory_importance(text, metadata))
            
            # Create task for entity extraction
            entity_tasks.append(self._extract_entities(text))
        
        # Wait for all tasks to complete
        importances = await asyncio.gather(*importance_tasks)
        entities_list = await asyncio.gather(*entity_tasks)
        
        # Store memories with calculated importances and extracted entities
        memory_ids = []
        
        for i, text in enumerate(texts):
            metadata = metadata_list[i] or {}
            if conversation_id and "conversation_id" not in metadata:
                metadata["conversation_id"] = conversation_id
                
            # Add importance to metadata
            metadata["importance"] = importances[i]
            
            # Generate a unique ID for the memory
            memory_id = str(uuid.uuid4())
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Add memory to vector store
            self.collection.add(
                documents=[text],
                metadatas=[{
                    "id": memory_id,
                    "timestamp": timestamp,
                    "importance": importances[i],
                    **metadata
                }],
                ids=[memory_id]
            )
            
            # Store memory in Neo4j with entities
            with self.driver.session() as session:
                session.execute_write(
                    self._create_memory_with_entities,
                    memory_id=memory_id,
                    text=text,
                    metadata={
                        "timestamp": timestamp,
                        "importance": importances[i],
                        **metadata
                    },
                    entities=entities_list[i]
                )
                
            memory_ids.append(memory_id)
            
        # Update context window with the new memories
        new_memories = []
        for i, memory_id in enumerate(memory_ids):
            new_memories.append({
                "id": memory_id,
                "text": texts[i],
                "timestamp": datetime.now().isoformat(),
                "importance": importances[i],
                "metadata": metadata_list[i] or {},
                "entities": entities_list[i]
            })
            
        self._update_context_window(new_memories)
            
        return memory_ids
        
    async def batch_recall_memories(self, queries: List[str], category: str = None, conversation_id: str = None, min_importance: float = 0.0, top_k: int = 5) -> List[List[Dict]]:
        """
        Recall memories for multiple queries in parallel.
        
        Args:
            queries: List of query strings
            category: Optional category to filter memories
            conversation_id: Optional conversation ID to filter memories
            min_importance: Minimum importance score for memories to be returned
            top_k: Maximum number of memories to return per query
            
        Returns:
            List of lists of memory dictionaries, one list per query
        """
        if not queries:
            return []
            
        # Process queries in parallel
        import asyncio
        
        # Create tasks for recalling memories
        tasks = []
        
        for query in queries:
            tasks.append(self.recall_memories(query, category, conversation_id, min_importance, top_k))
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        return results

    def generate_memory_graph_visualization(self, output_file: str = "memory_graph.html", max_nodes: int = 100):
        """
        Generate an interactive visualization of the memory graph.
        
        Args:
            output_file: Path to save the HTML visualization
            max_nodes: Maximum number of nodes to include in the visualization
            
        Returns:
            Path to the generated visualization file
        """
        try:
            import networkx as nx
            from pyvis.network import Network
            
            # Create a NetworkX graph
            G = nx.Graph()
            
            # Query Neo4j for nodes and relationships
            with self.driver.session() as session:
                # Get memories
                memory_query = f"""
                MATCH (m:Memory)
                RETURN m.id AS id, m.text AS text, m.importance AS importance, 
                       m.category AS category, toString(m.timestamp) AS timestamp
                ORDER BY m.importance DESC
                LIMIT {max_nodes // 2}
                """
                
                memory_result = session.run(memory_query)
                memories = [dict(record) for record in memory_result]
                
                if not memories:
                    print("No memories found in the database")
                    return None
                
                # Add memory nodes to graph
                for memory in memories:
                    # Truncate text for display
                    display_text = memory["text"][:50] + "..." if len(memory["text"]) > 50 else memory["text"]
                    
                    # Add node with attributes - convert all values to JSON-serializable types
                    G.add_node(
                        memory["id"],
                        label=display_text,
                        title=memory["text"],  # Full text on hover
                        group="memory",
                        importance=float(memory.get("importance", 0.5)),
                        category=str(memory.get("category", "general")),
                        timestamp=str(memory.get("timestamp", ""))
                    )
                
                # Get entities
                entity_query = f"""
                MATCH (e:Entity)<-[:CONTAINS]-(m:Memory)
                WHERE m.id IN $memory_ids
                RETURN e.id AS id, e.value AS value, e.type AS type, 
                       count(m) AS memory_count
                ORDER BY memory_count DESC
                LIMIT {max_nodes // 2}
                """
                
                entity_result = session.run(entity_query, memory_ids=[m["id"] for m in memories])
                entities = [dict(record) for record in entity_result]
                
                if not entities:
                    print("No entities found related to memories")
                    # Create a simple graph with just memories
                    net = Network(height="800px", width="100%", notebook=False, directed=False)
                    net.from_nx(G)
                    net.save_graph(output_file)
                    print(f"Memory graph visualization (memories only) saved to {output_file}")
                    return output_file
                
                # Add entity nodes to graph
                for entity in entities:
                    if "id" not in entity or not entity["id"]:
                        continue  # Skip entities without ID
                        
                    # Use value as label if available, otherwise use ID
                    label = str(entity.get("value", entity["id"]))
                    if not label:
                        continue  # Skip entities without label
                        
                    G.add_node(
                        entity["id"],
                        label=label,
                        title=f"{entity.get('type', 'Entity')}: {label}",
                        group=str(entity.get("type", "entity")).lower(),
                        memory_count=int(entity.get("memory_count", 1))
                    )
                
                # Get relationships between memories and entities
                rel_query = """
                MATCH (m:Memory)-[:CONTAINS]->(e:Entity)
                WHERE m.id IN $memory_ids AND e.id IN $entity_ids
                RETURN m.id AS memory_id, e.id AS entity_id
                """
                
                rel_result = session.run(
                    rel_query, 
                    memory_ids=[m["id"] for m in memories],
                    entity_ids=[e["id"] for e in entities]
                )
                
                # Add edges to graph
                for record in rel_result:
                    if record["memory_id"] and record["entity_id"]:
                        G.add_edge(record["memory_id"], record["entity_id"])
            
            # Create a PyVis network from the NetworkX graph
            net = Network(height="800px", width="100%", notebook=False, directed=False)
            
            # Set physics options for better visualization
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=250,
                spring_strength=0.001,
                damping=0.09
            )
            
            # Add the graph to the network
            net.from_nx(G)
            
            # Configure node appearance
            for node in net.nodes:
                if node.get("group") == "memory":
                    node["color"] = "#6929c4"  # Purple for memories
                    node["size"] = 15 + (float(node.get("importance", 0.5)) * 15)  # Size based on importance
                else:
                    # Color entities by type
                    entity_colors = {
                        "person": "#1192e8",   # Blue
                        "org": "#005d5d",      # Teal
                        "gpe": "#9f1853",      # Magenta
                        "loc": "#fa4d56",      # Red
                        "product": "#570408",  # Maroon
                        "event": "#198038",    # Green
                        "date": "#b28600",     # Yellow
                        "time": "#8a3800",     # Orange
                        "concept": "#a56eff",  # Purple
                        "topic": "#009d9a"     # Cyan
                    }
                    
                    node_group = str(node.get("group", "entity")).lower()
                    node["color"] = entity_colors.get(node_group, "#6f6f6f")  # Default gray
                    node["size"] = 10 + (int(node.get("memory_count", 1)) * 2)  # Size based on memory count
            
            # Save the visualization
            net.save_graph(output_file)
            
            print(f"Memory graph visualization saved to {output_file}")
            return output_file
            
        except ImportError:
            print("Error: Required packages not installed. Please install networkx and pyvis.")
            return None
        except Exception as e:
            print(f"Error generating memory graph visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def adapt_importance_thresholds(self):
        """
        Adapt importance thresholds based on memory usage patterns.
        
        This method analyzes memory access patterns and adjusts importance thresholds
        to better retain frequently accessed topics and categories.
        """
        try:
            # Get memory access statistics
            with self.driver.session() as session:
                # Get category statistics
                category_query = """
                MATCH (m:Memory)
                WHERE m.last_accessed IS NOT NULL
                WITH m.category AS category, 
                     count(*) AS total_count,
                     avg(m.importance) AS avg_importance,
                     sum(CASE WHEN m.references > 0 THEN 1 ELSE 0 END) AS referenced_count
                RETURN category, total_count, avg_importance, referenced_count,
                       toFloat(referenced_count) / total_count AS reference_ratio
                ORDER BY reference_ratio DESC
                """
                
                category_result = session.run(category_query)
                category_stats = [dict(record) for record in category_result]
                
                # Get entity type statistics
                entity_query = """
                MATCH (e:Entity)<-[:CONTAINS]-(m:Memory)
                WHERE m.last_accessed IS NOT NULL
                WITH e.type AS entity_type, 
                     count(DISTINCT e) AS entity_count,
                     count(DISTINCT m) AS memory_count,
                     avg(m.importance) AS avg_importance
                RETURN entity_type, entity_count, memory_count, avg_importance
                ORDER BY memory_count DESC
                """
                
                entity_result = session.run(entity_query)
                entity_stats = [dict(record) for record in entity_result]
                
                # Get time-based access patterns
                time_query = """
                MATCH (m:Memory)
                WHERE m.last_accessed IS NOT NULL
                WITH m,
                     duration.inDays(datetime(m.timestamp), datetime(m.last_accessed)).days AS age_at_access
                RETURN avg(age_at_access) AS avg_age_at_access,
                       percentileCont(age_at_access, 0.5) AS median_age_at_access,
                       percentileCont(age_at_access, 0.9) AS p90_age_at_access
                """
                
                time_result = session.run(time_query)
                time_stats = dict(time_result.single()) if time_result.peek() else {
                    "avg_age_at_access": 0,
                    "median_age_at_access": 0,
                    "p90_age_at_access": 0
                }
            
            # Calculate adaptive thresholds based on statistics
            
            # 1. Category-based thresholds
            category_thresholds = {}
            for stat in category_stats:
                category = stat["category"]
                reference_ratio = stat["reference_ratio"]
                
                # Higher reference ratio = lower threshold (keep more memories)
                category_thresholds[category] = max(0.2, min(0.8, 0.6 - (reference_ratio * 0.5)))
            
            # 2. Entity type-based thresholds
            entity_thresholds = {}
            if entity_stats:
                max_memory_count = max(stat["memory_count"] for stat in entity_stats)
                
                for stat in entity_stats:
                    entity_type = stat["entity_type"]
                    memory_ratio = stat["memory_count"] / max_memory_count if max_memory_count > 0 else 0
                    
                    # Higher memory ratio = lower threshold (keep more memories)
                    entity_thresholds[entity_type] = max(0.2, min(0.8, 0.6 - (memory_ratio * 0.4)))
            
            # 3. Time-based threshold adjustment
            time_factor = 1.0
            if time_stats["avg_age_at_access"] > 0:
                # If memories are typically accessed long after creation, reduce decay rate
                avg_age = time_stats["avg_age_at_access"]
                time_factor = max(0.7, min(1.3, 1.0 + (avg_age / 30) * 0.1))  # +/- 30% based on avg age
            
            # Update memory config with adaptive thresholds
            self.adaptive_thresholds = {
                "category": category_thresholds,
                "entity": entity_thresholds,
                "time_factor": time_factor,
                "last_updated": datetime.now().isoformat()
            }
            
            # Adjust global decay factor based on time patterns
            self.config.memory_decay_factor = min(0.99, max(0.9, self.config.memory_decay_factor * time_factor))
            
            print(f"Adapted importance thresholds based on usage patterns")
            print(f"New decay factor: {self.config.memory_decay_factor}")
            
            return self.adaptive_thresholds
            
        except Exception as e:
            print(f"Error adapting importance thresholds: {e}")
            return None
    
    def get_adaptive_threshold(self, memory: Dict) -> float:
        """
        Get the adaptive importance threshold for a specific memory.
        
        Args:
            memory: Memory dictionary with category and entities
            
        Returns:
            Adaptive importance threshold for the memory
        """
        if not hasattr(self, 'adaptive_thresholds'):
            return self.config.importance_threshold
            
        # Start with base threshold
        threshold = self.config.importance_threshold
        
        # Adjust based on category
        category = memory.get("category", "general")
        if category in self.adaptive_thresholds.get("category", {}):
            threshold = self.adaptive_thresholds["category"][category]
        
        # Adjust based on entities
        if "entities" in memory:
            entity_thresholds = []
            for entity_type, entities in memory["entities"].items():
                if entity_type in self.adaptive_thresholds.get("entity", {}):
                    entity_thresholds.append(self.adaptive_thresholds["entity"][entity_type])
            
            # Use the lowest entity threshold if any
            if entity_thresholds:
                threshold = min(threshold, min(entity_thresholds))
        
        return threshold

    def _robust_json_parse(self, text: str) -> Dict:
        """
        Robustly parse JSON from LLM responses, handling common formatting issues.
        
        Args:
            text: Text containing JSON to parse
            
        Returns:
            Parsed JSON as dictionary, or empty dict if parsing fails
        """
        import re
        import json
        
        # Strip any markdown code block markers
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return {}
            
        json_str = json_match.group(0)
        
        # Fix common JSON formatting issues
        
        # 1. Remove trailing commas in objects
        json_str = re.sub(r',\s*\}', '}', json_str)
        
        # 2. Remove trailing commas in arrays
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # 3. Ensure property names are double-quoted
        # This regex finds property names that aren't properly quoted
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 4. Fix single quotes to double quotes (but not inside already double-quoted strings)
        # This is complex, so we'll use a simpler approach that works for most cases
        in_string = False
        result = []
        for char in json_str:
            if char == '"':
                in_string = not in_string
            elif char == "'" and not in_string:
                char = '"'
            result.append(char)
        json_str = ''.join(result)
        
        # 5. Handle unquoted string values
        # This regex finds unquoted string values and quotes them
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            
            # Last resort: try to fix the specific error
            if "Expecting ',' delimiter" in str(e):
                # Try to insert a comma at the position indicated by the error
                pos = int(re.search(r'char (\d+)', str(e)).group(1))
                json_str = json_str[:pos] + ',' + json_str[pos:]
                try:
                    return json.loads(json_str)
                except:
                    pass
                    
            return {}

async def main():
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get API key from environment
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment variables")
            return
            
        # Initialize Groq provider
        groq_provider = GroqProvider(
            api_key=api_key,
            model_name="llama3-8b-8192"
        )
        
        # Initialize memory module
        memory_module = EpisodicMemoryModule(
            llm_provider=groq_provider,
            collection_name="demo_memory",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        print("\n===== Episodic Memory Module Demo =====\n")
        
        # 1. Store some memories
        print("1. Storing memories...")
        
        # Store memories in parallel
        texts = [
            "Python is a versatile programming language used for web development, data analysis, and AI.",
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "The Ebbinghaus forgetting curve describes how memory retention decreases over time.",
            "Spaced repetition is a learning technique that incorporates increasing intervals of time between reviews."
        ]
        
        metadata_list = [
            {"category": "programming", "priority": "HIGH"},
            {"category": "ai", "priority": "MEDIUM"},
            {"category": "ai", "priority": "MEDIUM"},
            {"category": "psychology", "priority": "LOW"},
            {"category": "learning", "priority": "MEDIUM"}
        ]
        
        memory_ids = await memory_module.batch_store_memories(texts, metadata_list)
        print(f"Stored {len(memory_ids)} memories")
        
        # 2. Recall memories
        print("\n2. Recalling memories...")
        query = "How does AI work?"
        print(f"Query: {query}")
        
        memories = await memory_module.recall_memories(query, top_k=3)
        print(f"Found {len(memories)} related memories:")
        for i, mem in enumerate(memories):
            print(f"  {i+1}. {mem['text']}")
            print(f"     Importance: {mem.get('importance', 'N/A'):.2f}")
            print(f"     Relevance: {mem.get('relevance', 'N/A'):.2f}")
        
        # 3. Batch recall for multiple queries
        print("\n3. Batch recall for multiple queries...")
        queries = [
            "Tell me about programming languages",
            "How does memory work in humans?"
        ]
        
        batch_results = await memory_module.batch_recall_memories(queries, top_k=2)
        for i, (query, results) in enumerate(zip(queries, batch_results)):
            print(f"\nQuery {i+1}: {query}")
            print(f"Found {len(results)} related memories:")
            for j, mem in enumerate(results):
                print(f"  {j+1}. {mem['text']}")
                print(f"     Relevance: {mem.get('relevance', 'N/A'):.2f}")
        
        # 4. Analyze entity relationships
        print("\n4. Analyzing entity relationships...")
        relationships = memory_module.get_entity_relationships()
        if relationships:
            print(f"Found {len(relationships)} entity relationships")
            for i, rel in enumerate(relationships[:2]):  # Show top 2 relationships
                print(f"\nRelationship {i+1}: {rel['entity1']} <-> {rel['entity2']}")
                print(f"Shared contexts: {len(rel['shared_contexts'])}")
                print(f"Example context: {rel['shared_contexts'][0] if rel['shared_contexts'] else 'None'}")
        else:
            print("No significant entity relationships found")

        # 5. Generate memory graph visualization
        print("\n5. Generating memory graph visualization...")
        viz_file = memory_module.generate_memory_graph_visualization("memory_graph.html")
        if viz_file:
            print(f"Memory graph visualization saved to {viz_file}")
        
        # 6. Adapt importance thresholds
        print("\n6. Adapting importance thresholds...")
        thresholds = await memory_module.adapt_importance_thresholds()
        if thresholds:
            print("Adapted thresholds:")
            for category, threshold in thresholds.get("category", {}).items():
                print(f"  Category '{category}': {threshold:.2f}")
        
        # 7. Apply memory decay
        print("\n7. Applying memory decay...")
        await memory_module.cleanup_memories()
        
        # 8. Get memory stats
        print("\n8. Memory statistics:")
        stats = memory_module.get_memory_stats()
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Total entities: {stats.get('total_entities', 0)}")
        print(f"  Average importance: {stats.get('avg_importance', 0):.2f}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
