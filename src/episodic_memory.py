import re
import os
import sys
import uuid
import json
import time
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
from datetime import datetime, timedelta, time
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Union, Any
import math
import random

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
        password: str = "password",
        min_delay: float = 2.0,
        max_delay: float = 30.0,
        max_retries: int = 3
    ):
        # Initialize maintenance_tasks as an empty list to ensure it always exists
        self.maintenance_tasks = []
        
        # Store rate limiting parameters
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        
        # Neo4j rate limit tracking
        self.neo4j_remaining_queries = 1000
        self.neo4j_reset_time = time.time() + 3600
        
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
        
        # Set the LLM provider
        self.llm = llm_provider
        
        # Initialize ChromaDB
        try:
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
            
            # Set memory_collection alias for the collection
            self.memory_collection = self.collection
            
            # Initialize common English stopwords
            self.stopwords = {
                "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                "when", "where", "how", "who", "which", "this", "that", "these", "those",
                "then", "just", "so", "than", "such", "both", "through", "about", "for",
                "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "do", "does", "did", "to", "from", "by", "on", "at", "in", "with", "of"
            }
            
            try:
                # Try to import NLTK stopwords if available for a more comprehensive list
                import nltk
                from nltk.corpus import stopwords
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
            except (ImportError, LookupError):
                # Keep the default stopwords if NLTK is not available
                pass
            
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
            try:
                self.maintenance_tasks = [
                    asyncio.create_task(self._run_periodic_decay()),
                    asyncio.create_task(self._run_periodic_cleanup())
                ]
            except Exception as e:
                print(f"Warning: Could not initialize maintenance tasks: {e}")
                # Ensure maintenance_tasks is at least an empty list
                self.maintenance_tasks = []

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

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
        last_accessed: $timestamp,
        metadata: $metadata
        })
        RETURN m
        """
            
        # Extract metadata with defaults
        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        importance = metadata.get("importance", 0.5)
        category = metadata.get("category", "general")
        conversation_id = metadata.get("conversation_id", "default")
        
        # Serialize metadata to JSON string to avoid Neo4j type errors
        metadata_json = json.dumps(metadata)
        
        # Create memory node
        tx.run(
            memory_query,
            id=memory_id,
            text=text,
            timestamp=timestamp,
            importance=importance,
            category=category,
            conversation_id=conversation_id,
            metadata=metadata_json
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
        Recall memories related to a query with optimized performance and relevance.
        
        Args:
            query: The query to search for related memories
            category: Optional category to filter memories
            conversation_id: Optional conversation ID to filter memories
            min_importance: Minimum importance score for memories to be returned
            top_k: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries with their text and metadata
        """
        if not query or not query.strip():
            return []
        
        # Check cache first for identical queries to improve performance
        cache_key = f"{query}_{category}_{conversation_id}_{min_importance}_{top_k}"
        if hasattr(self, '_memory_cache') and cache_key in self._memory_cache:
            # Update access time for cached memories using robust query execution
            try:
                memory_ids = [m['id'] for m in self._memory_cache[cache_key]]
                if memory_ids:
                    def update_access_times(session, memory_ids):
                        for memory_id in memory_ids:
                            session.execute_write(self._update_memory_access_tx, memory_id)
                        return True
                        
                    await self._execute_neo4j_query_with_retry(update_access_times, memory_ids)
            except Exception as e:
                print(f"Warning: Failed to update memory access times: {e}")
                
            return self._memory_cache[cache_key]
        
        # Initialize memory cache if needed
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
        
        try:
            # Build where clause for filtering
            where_clause = self._build_where_clause(category, conversation_id, min_importance)
            
            # Extract keywords from the query for better context matching
            keywords, entity_names, is_question = self._analyze_query(query)
            
            # Empty results as default
            memories = []
            
            # Collect memories from multiple sources
            try:
                # 1. Vector search for semantic similarity
                vector_memories = await self._get_vector_memories(query, where_clause, top_k * 3)
                
                # 2. Keyword-based search for explicit mentions
                def get_keyword_memories(session, keywords, where_clause, top_k):
                    return session.execute_read(
                        self._get_keyword_memories_tx, 
                        keywords, 
                        where_clause, 
                        top_k * 2
                    )
                    
                keyword_memories = await self._execute_neo4j_query_with_retry(
                    get_keyword_memories,
                    keywords,
                    where_clause,
                    top_k * 2
                )
                
                # 3. Entity-based search if entities were detected in the query
                entity_memories = []
                if entity_names:
                    def get_entity_memories(session, entity_names, where_clause, top_k):
                        return session.execute_read(
                            self._get_entity_related_memories_tx, 
                            entity_names, 
                            where_clause, 
                            top_k * 2
                        )
                        
                    entity_memories = await self._execute_neo4j_query_with_retry(
                        get_entity_memories,
                        entity_names,
                        where_clause,
                        top_k * 2
                    )
                
                # Combine memories from different sources with de-duplication
                combined_memories = {}
                
                for memory in vector_memories:
                    memory_id = memory.get("id")
                    if memory_id:
                        memory["source"] = "vector"
                        memory["vector_score"] = memory.get("score", 0)
                        combined_memories[memory_id] = memory
                
                for memory in keyword_memories:
                    memory_id = memory.get("id")
                    if memory_id and memory_id in combined_memories:
                        combined_memories[memory_id]["keyword_match"] = True
                        combined_memories[memory_id]["source"] = combined_memories[memory_id].get("source", "") + "+keyword"
                    elif memory_id:
                        memory["source"] = "keyword"
                        memory["keyword_match"] = True
                        combined_memories[memory_id] = memory
                
                for memory in entity_memories:
                    memory_id = memory.get("id")
                    if memory_id and memory_id in combined_memories:
                        combined_memories[memory_id]["entity_match"] = True
                        combined_memories[memory_id]["entity_score"] = memory.get("score", 0)
                        combined_memories[memory_id]["source"] = combined_memories[memory_id].get("source", "") + "+entity"
                    elif memory_id:
                        memory["source"] = "entity"
                        memory["entity_match"] = True
                        memory["entity_score"] = memory.get("score", 0)
                        combined_memories[memory_id] = memory
                
                # Convert back to list
                memories = list(combined_memories.values())
                
            except Exception as e:
                print(f"Error retrieving memories: {e}")
                import traceback
                traceback.print_exc()
                memories = []
            
            # Enhanced scoring - dynamically adjust weights based on query type
            memories = await self._rank_memories(query, keywords, entity_names, is_question, memories)
            
            # Cache the results for future identical queries
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
                
            self._memory_cache[cache_key] = memories
            
            # Limit cache size to prevent memory issues
            if len(self._memory_cache) > 100:
                # Remove random entries to keep cache size reasonable
                keys_to_remove = random.sample(list(self._memory_cache.keys()), min(50, len(self._memory_cache)))
                for key in keys_to_remove:
                    del self._memory_cache[key]
            
            return memories
            
        except Exception as e:
            print(f"Error recalling memories: {e}")
            import traceback
            traceback.print_exc()
            return []

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
        """Get memories related to a query from Neo4j, applying where clause filters"""
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

    def _get_keyword_memories_tx(self, tx, keywords, where_clause, top_k=5):
        """
        Retrieve memories based on keyword matching
        
        Args:
            tx: Neo4j transaction
            keywords: Set of keywords to match
            where_clause: Additional filtering conditions
            top_k: Maximum number of results to return
            
        Returns:
            List of memory dictionaries with matched keywords
        """
        # Prepare a LIKE clause for each keyword
        keyword_clauses = []
        params = where_clause.copy()
        
        for i, keyword in enumerate(keywords):
            param_name = f"keyword_{i}"
            keyword_clauses.append(f"m.text CONTAINS ${param_name}")
            params[param_name] = keyword
        
        # If we have no keywords, return empty
        if not keyword_clauses:
            return []
        
        # Build the where conditions from the where_clause dict
        conditions = []
        for key, value in where_clause.items():
            if key.startswith("keyword_"):
                continue  # Skip keyword parameters
            
            if isinstance(value, list):
                conditions.append(f"m.{key} IN ${key}")
            else:
                conditions.append(f"m.{key} = ${key}")
        
        # Add importance threshold
        if "min_importance" in where_clause:
            conditions.append(f"m.importance >= $min_importance")
        
        # Add the keyword conditions with OR between them
        keyword_condition = " OR ".join(keyword_clauses)
        conditions.append(f"({keyword_condition})")
        
        # Build the full where clause
        where_condition = " AND ".join(conditions)
        
        # Construct the Cypher query
        query = f"""
        MATCH (m:Memory)
        WHERE {where_condition}
        RETURN m.id as id, m.text as text, m.importance as importance, 
               m.category as category, m.timestamp as timestamp,
               m.metadata as metadata, m.references as references
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT {top_k}
        """
        
        # Execute the query
        result = tx.run(query, **params)
        
        # Process and return the results
        memories = []
        for record in result:
            # Convert Neo4j record to dict
            memory = dict(record)
            
            # Deserialize metadata from JSON string if needed
            if "metadata" in memory and isinstance(memory["metadata"], str):
                try:
                    memory["metadata"] = json.loads(memory["metadata"])
                except (json.JSONDecodeError, TypeError):
                    memory["metadata"] = {}
            
            # Convert timestamp from string to datetime if needed
            if isinstance(memory.get("timestamp"), str):
                try:
                    memory["timestamp"] = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If conversion fails, keep the original
                    pass
                
            # Handle case where metadata is missing or null in database
            if "metadata" not in memory or memory["metadata"] is None:
                memory["metadata"] = {}
                
            memories.append(memory)
        
        return memories

    def _get_entity_related_memories_tx(self, tx, entity_names, where_clause, top_k=5):
        """
        Retrieve memories related to specific entities through graph relationships
        
        Args:
            tx: Neo4j transaction
            entity_names: List of entity names to match
            where_clause: Additional filtering conditions
            top_k: Maximum number of results to return
            
        Returns:
            List of memory dictionaries related to the entities
        """
        # Prepare parameters
        params = where_clause.copy()
        params["entity_names"] = entity_names
        
        # Build the where conditions from the where_clause dict
        conditions = []
        for key, value in where_clause.items():
            if key == "entity_names":
                continue  # Skip entity names parameter
                
            if isinstance(value, list):
                conditions.append(f"m.{key} IN ${key}")
            else:
                conditions.append(f"m.{key} = ${key}")
        
        # Add importance threshold
        if "min_importance" in where_clause:
            conditions.append(f"m.importance >= $min_importance")
        
        # Combine entity name condition with other conditions
        combined_conditions = ["e.name IN $entity_names"]
        combined_conditions.extend(conditions)
        
        # Build the full where clause 
        where_condition = "WHERE " + " AND ".join(combined_conditions)
        
        # Construct the Cypher query to find memories connected to the entities
        query = f"""
        MATCH (m:Memory)-[r:CONTAINS]->(e:Entity)
        {where_condition}
        WITH m, count(e) as entity_matches
        RETURN m.id as id, m.text as text, m.importance as importance, 
               m.category as category, m.timestamp as timestamp,
               m.metadata as metadata, m.references as references,
               entity_matches as score
        ORDER BY entity_matches DESC, m.importance DESC
        LIMIT {top_k}
        """
        
        # Execute the query
        result = tx.run(query, **params)
        
        # Process and return the results
        memories = []
        for record in result:
            # Convert Neo4j record to dict
            memory = dict(record)
            
            # Deserialize metadata from JSON string if needed
            if "metadata" in memory and isinstance(memory["metadata"], str):
                try:
                    memory["metadata"] = json.loads(memory["metadata"])
                except (json.JSONDecodeError, TypeError):
                    memory["metadata"] = {}
            
            # Normalize the score to 0-1 range
            if "score" in memory:
                memory["score"] = min(1.0, memory["score"] / len(entity_names))
            
            # Convert timestamp from string to datetime if needed
            if isinstance(memory.get("timestamp"), str):
                try:
                    memory["timestamp"] = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If conversion fails, keep the original
                    pass
                
            memories.append(memory)
        
        return memories

    def _get_memory_with_context_tx(self, tx, memory_id):
        """Get a memory with its connected entities and related metadata"""
        query = """
        MATCH (m:Memory {id: $memory_id})
        OPTIONAL MATCH (m)-[r:CONTAINS]->(e:Entity)
        RETURN m.id as id, m.text as text, m.importance as importance, 
               m.category as category, m.timestamp as timestamp,
               m.metadata as metadata, m.references as references,
               collect(e.name) as entities
        """
        
        record = tx.run(query, memory_id=memory_id).single()
        if not record:
            return None
            
        # Convert Neo4j record to dict
        memory = dict(record)
        
        # Handle case where metadata is missing or null in database
        if "metadata" not in memory or memory["metadata"] is None:
            memory["metadata"] = {}
        else:
            # Deserialize metadata from JSON string if it's stored as a string
            try:
                if isinstance(memory["metadata"], str):
                    memory["metadata"] = json.loads(memory["metadata"])
            except (json.JSONDecodeError, TypeError):
                # If deserialization fails, keep as is or use empty dict
                memory["metadata"] = {}
            
        # Ensure entities is at least an empty list
        if "entities" not in memory or memory["entities"] is None:
            memory["entities"] = []
            
        # Convert Neo4j timestamp to Python datetime if needed
        if "timestamp" in memory and isinstance(memory["timestamp"], str):
            try:
                memory["timestamp"] = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                memory["timestamp"] = datetime.now()
        
        return memory

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
        Extract entities from text using a combination of NLP techniques with optimized performance.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of entity types to lists of entity values
        """
        try:
            # Cache check - use simple hash of input text for cache key
            cache_key = hash(text)
            if hasattr(self, '_entity_cache') and cache_key in self._entity_cache:
                return self._entity_cache[cache_key]
            
            # Initialize entity cache if needed
            if not hasattr(self, '_entity_cache'):
                self._entity_cache = {}
                
            # Use spaCy for initial entity extraction (faster pipeline)
            doc = self.nlp(text)
            
            # Process in a single pass for better efficiency
            entities = {}
            
            # Map for quick entity category lookup
            category_map = {}
            for category, labels in self.entity_config.items():
                for label in labels:
                    category_map[label] = category
            
            # Process spaCy entities directly into the correct categories
            for ent in doc.ents:
                # Map to our category or use original label
                category = category_map.get(ent.label_, ent.label_)
                
                if category not in entities:
                    entities[category] = []
                    
                # Clean and normalize text
                clean_text = ent.text.strip()
                if clean_text and clean_text.lower() not in [e.lower() for e in entities[category]]:
                    entities[category].append(clean_text)
            
            # Extract named entities specifically - prioritize these for better recall
            if "PERSON" not in entities:
                entities["PERSON"] = []
            if "ORG" not in entities:
                entities["ORG"] = []
            if "GPE" not in entities:
                entities["GPE"] = []
                
            # Enhanced pattern matching for common entities like emails, URLs, dates, proper nouns
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(email_pattern, text):
                if "EMAIL" not in entities:
                    entities["EMAIL"] = []
                email = match.group(0)
                if email not in entities["EMAIL"]:
                    entities["EMAIL"].append(email)
            
            # URL pattern
            url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            for match in re.finditer(url_pattern, text):
                if "URL" not in entities:
                    entities["URL"] = []
                url = match.group(0)
                if url not in entities["URL"]:
                    entities["URL"].append(url)
            
            # Date pattern
            date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s\d{2,4})\b'
            for match in re.finditer(date_pattern, text, re.IGNORECASE):
                if "DATE" not in entities:
                    entities["DATE"] = []
                date = match.group(0)
                if date not in entities["DATE"]:
                    entities["DATE"].append(date)
            
            # Quick pattern matching for technical entities
            tech_matches = set()
            for pattern in self.tech_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    tech_matches.add(match.group(0))
            
            if tech_matches:
                if "TECH" not in entities:
                    entities["TECH"] = []
                for match in tech_matches:
                    if match not in entities["TECH"]:
                        entities["TECH"].append(match)
            
            # Extract capitalized multi-word phrases as potential entities
            cap_pattern = r'\b(?:[A-Z][a-z]+\s+){1,5}[A-Z][a-z]+\b'
            for match in re.finditer(cap_pattern, text):
                phrase = match.group(0)
                # Check if this phrase is already covered by an existing entity
                is_covered = False
                for ent_list in entities.values():
                    if any(phrase in entity or entity in phrase for entity in ent_list):
                        is_covered = True
                        break
                
                if not is_covered:
                    if "MISC" not in entities:
                        entities["MISC"] = []
                    if phrase not in entities["MISC"]:
                        entities["MISC"].append(phrase)
            
            # Cache the result
            self._entity_cache[cache_key] = entities
            
            # Limit cache size to prevent memory issues
            if len(self._entity_cache) > 1000:
                # Remove 200 random entries
                keys_to_remove = random.sample(list(self._entity_cache.keys()), min(200, len(self._entity_cache)))
                for key in keys_to_remove:
                    del self._entity_cache[key]
                    
            return entities
        
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}

    def _update_context_window(self, new_memories: List[Dict]):
        """
        Update context window with efficient memory management and improved relevance weighing
        
        Args:
            new_memories: List of new memories to consider for the context window
        """
        try:
            if not isinstance(new_memories, list):
                raise ValueError("new_memories must be a list")
            
            # Ensure all memories have required fields to prevent NoneType errors
            validated_memories = []
            for memory in new_memories:
                if not memory:
                    continue
                    
                # Ensure all required fields exist
                if "id" not in memory or "text" not in memory:
                    continue
                    
                # Ensure metadata is a dictionary
                if "metadata" not in memory or memory["metadata"] is None:
                    memory["metadata"] = {}
                
                # Ensure other fields have default values if missing
                memory["importance"] = memory.get("importance", 0.5)
                memory["category"] = memory.get("category", "general")
                memory["references"] = memory.get("references", 0)
                memory["timestamp"] = memory.get("timestamp", datetime.now())
                memory["last_accessed"] = memory.get("last_accessed", datetime.now())
                
                validated_memories.append(memory)
            
            # Replace new_memories with validated list
            new_memories = validated_memories
            
            current_time = datetime.now()
            
            try:
                # Stage 1: Process existing context window
                # Update importance based on decay and time since last access
                for memory in self.context_window:
                    if "last_accessed" in memory:
                        # Calculate time-based decay
                        time_diff = (current_time - memory["last_accessed"]).total_seconds() / 3600
                        # Use more gradual decay for recently accessed memories
                        if time_diff < 24:  # Less than a day
                            decay = max(0.9, self.config.memory_decay_factor ** (time_diff / 72))
                        else:
                            decay = max(0.5, self.config.memory_decay_factor ** (time_diff / 24))
                        
                        # Apply reference count bonus
                        reference_bonus = min(0.2, (memory.get("references", 0) * 0.05))
                        
                        # Apply decay with reference bonus
                        memory["importance"] = min(1.0, memory.get("importance", 0.5) * decay + reference_bonus)
                
                # Stage 2: Prepare combined memory list and deduplicate
                # Create a combined list of all memories (existing and new)
                all_memories = self.context_window.copy()
                
                # Create a set of existing memory texts (lowercase for case-insensitive comparison)
                existing_texts = {memory.get("text", "").strip().lower() for memory in all_memories}
                
                # Add new memories, avoiding duplicates
                for memory in new_memories:
                    if not isinstance(memory, dict) or "text" not in memory:
                        print(f"Skipping invalid memory format: {memory}")
                        continue
                    
                    text_key = memory.get("text", "").strip().lower()
                    if not text_key:
                        continue
                        
                    # Check for duplicates or near-duplicates
                    is_duplicate = False
                    
                    # Exact duplicate check
                    if text_key in existing_texts:
                        is_duplicate = True
                        # Update existing memory rather than add a new one
                        for existing_memory in all_memories:
                            if existing_memory.get("text", "").strip().lower() == text_key:
                                # Boost importance when a memory is recalled again
                                existing_memory["importance"] = min(1.0, existing_memory.get("importance", 0.5) + 0.1)
                                existing_memory["last_accessed"] = current_time
                                existing_memory["references"] = existing_memory.get("references", 0) + 1
                                break
                    
                    # Add new memory if it's not a duplicate
                    if not is_duplicate:
                        memory_copy = memory.copy()
                        memory_copy["last_accessed"] = current_time
                        memory_copy["references"] = memory_copy.get("references", 0)
                        all_memories.append(memory_copy)
                        existing_texts.add(text_key)
                
                # Stage 3: Filter and prioritize memories
                # Calculate a combined score for each memory using multiple factors
                for memory in all_memories:
                    # Base importance from memory
                    base_importance = memory.get("importance", 0.5)
                    
                    # Recency factor - boost more recently accessed memories
                    recency_score = 0.5
                    if "last_accessed" in memory:
                        age_hours = (current_time - memory["last_accessed"]).total_seconds() / 3600
                        recency_score = max(0.1, min(1.0, math.exp(-age_hours / 48)))  # 48-hour half-life
                    
                    # Reference count factor - boost more frequently referenced memories
                    reference_score = 0.5
                    if "references" in memory:
                        reference_score = min(1.0, 0.5 + (memory["references"] * 0.1))
                    
                    # Relevance factor - if provided from search results
                    relevance_score = memory.get("final_score", memory.get("score", 0.5))
                    
                    # Calculate combined prioritization score with weighted factors
                    # Heavier weight on relevance and importance
                    memory["priority_score"] = (
                        (base_importance * 0.4) +
                        (recency_score * 0.2) +
                        (reference_score * 0.15) +
                        (relevance_score * 0.25)
                    )
                
                # Sort memories by priority score (descending) and limit to max_context_items
                self.context_window = sorted(
                    all_memories,
                    key=lambda x: x.get("priority_score", 0),
                    reverse=True
                )[:self.config.max_context_items]
                
                # Ensure we have a diverse set of memories by category if possible
                if len(self.context_window) > 5:
                    # Count memories by category
                    category_counts = {}
                    for memory in self.context_window:
                        category = memory.get("category", "general")
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    # Identify overrepresented categories
                    avg_count = len(self.context_window) / max(1, len(category_counts))
                    overrepresented = [cat for cat, count in category_counts.items() 
                                      if count > avg_count * 1.5 and count > 2]
                    
                    # If we have overrepresented categories, replace some with memories from underrepresented ones
                    if overrepresented and len(all_memories) > len(self.context_window):
                        # Memories already in the window
                        selected_ids = {mem.get("id") for mem in self.context_window}
                        
                        # Find memories from underrepresented categories
                        underrepresented_memories = [
                            mem for mem in all_memories 
                            if mem.get("id") not in selected_ids and
                            mem.get("category", "general") not in overrepresented
                        ]
                        
                        # Sort by priority
                        underrepresented_memories.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
                        
                        # Replace some memories from overrepresented categories
                        for cat in overrepresented:
                            # Find memories from this category
                            cat_memories = [mem for mem in self.context_window 
                                           if mem.get("category", "general") == cat]
                            
                            # Keep the top ones, replace others
                            to_replace = cat_memories[max(1, len(cat_memories) // 2):]
                            
                            for i, mem in enumerate(to_replace):
                                if i < len(underrepresented_memories):
                                    # Find index in context window
                                    idx = next((j for j, m in enumerate(self.context_window) 
                                              if m.get("id") == mem.get("id")), None)
                                    if idx is not None:
                                        # Replace with an underrepresented memory
                                        self.context_window[idx] = underrepresented_memories[i]
            
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
            # Cancel maintenance tasks if they exist
            if hasattr(self, 'maintenance_tasks'):
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
        Calculate the importance of a memory based on its content and metadata using
        an enhanced algorithm with multiple heuristics.
        
        Args:
            text: The memory text
            metadata: Optional metadata associated with the memory
            
        Returns:
            Importance score between 0 and 1
        """
        if not text or not text.strip():
            return 0.3  # Default low importance for empty or blank text
        
        # Start with base importance from metadata or default
        base_importance = 0.5
        if metadata and 'priority' in metadata:
            priority = metadata['priority']
            if isinstance(priority, str) and priority in self.config.priority_levels:
                base_importance = self.config.priority_levels[priority]
            elif isinstance(priority, (int, float)) and 0 <= priority <= 1:
                base_importance = priority
        
        # Use LLM to assess importance if available
        llm_importance = None
        if self.llm_provider:
            try:
                # Enhanced prompt with specific evaluation criteria
                prompt = f"""
                Rate the importance of the following information for long-term memory storage.
                
                TEXT: "{text}"
                
                Use these specific criteria:
                1. Information density: Does it contain many facts, details, or concepts?
                2. Uniqueness: How unique or rare is this information?
                3. Future relevance: How likely is this information to be useful in future conversations?
                4. Specificity: Does it contain specific details rather than generalities?
                5. Emotional/Personal significance: Does it contain emotionally relevant or personal details?
                
                For each criterion, score from 0.0 to 1.0, then provide a weighted average.
                
                Score each of the 5 criteria individually, then calculate a final score.
                Return ONLY the final numeric score between 0.0 and 1.0, nothing else.
                """
                
                response = await self.llm_provider.generate_text(prompt, max_tokens=50)
                # Extract the numeric value from the response
                match = re.search(r'(\d+\.\d+|\d+)', response)
                if match:
                    llm_importance = float(match.group(0))
                    # Ensure the value is between 0 and 1
                    llm_importance = max(0.0, min(1.0, llm_importance))
            except Exception as e:
                print(f"Error calculating memory importance with LLM: {e}")
        
        # Enhanced heuristic scoring when LLM is not available or fails
        # Create a list of heuristic scores
        heuristic_scores = []
        
        # 1. Length factor - longer text might contain more information
        normalized_length = min(len(text) / 500, 1.0)  # Normalize to max of 1.0
        length_factor = 0.3 + (normalized_length * 0.7)  # Scale between 0.3 and 1.0
        heuristic_scores.append(length_factor)
        
        # 2. Entity density - more entities might indicate more important information
        try:
            # Extract entities
            entities = {}
            try:
                doc = self.nlp(text)
                entities = {ent.label_: ent.text for ent in doc.ents}
            except Exception:
                pass
            
            # Calculate normalized entity density
            entity_count = len(entities)
            words = text.split()
            word_count = len(words)
            
            if word_count > 0:
                entity_density = min(entity_count / (word_count / 10), 1.0)  # Expect 1 entity per 10 words
            else:
                entity_density = 0.0
                
            heuristic_scores.append(entity_density)
        except Exception:
            heuristic_scores.append(0.0)
        
        # 3. Keyword importance - check for important keywords
        important_keywords = [
            "remember", "important", "critical", "essential", "key", "vital",
            "significant", "crucial", "necessary", "remember", "note", "attention",
            "password", "secret", "confidential", "private", "personal", "security"
        ]
        
        keyword_count = sum(1 for keyword in important_keywords if keyword.lower() in text.lower())
        keyword_factor = min(keyword_count / 3, 1.0)  # Normalize to max of 1.0
        heuristic_scores.append(keyword_factor)
        
        # 4. Question factor - text containing questions might be more important
        question_factor = 0.0
        if '?' in text:
            question_count = text.count('?')
            question_factor = min(question_count / 2, 1.0)  # Normalize to max of 1.0
        heuristic_scores.append(question_factor)
        
        # 5. Number density - text with numbers might contain factual information
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        number_density = min(len(numbers) / 5, 1.0)  # Normalize to max of 1.0
        heuristic_scores.append(number_density)
        
        # 6. Semantic diversity - text with diverse vocabulary might be more important
        unique_words = set(word.lower() for word in re.findall(r'\b\w+\b', text))
        total_words = len(re.findall(r'\b\w+\b', text))
        if total_words > 0:
            lexical_diversity = min(len(unique_words) / total_words * 2, 1.0)  # Scale up to favor diversity
        else:
            lexical_diversity = 0.0
        heuristic_scores.append(lexical_diversity)
        
        # 7. Recency factor - more recent memories might be more important (if timestamp available)
        recency_factor = 0.7  # Default moderate boost for new memories
        if metadata and 'timestamp' in metadata:
            try:
                timestamp = metadata['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Calculate hours since creation
                hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
                
                # Decay factor based on recency (stronger in first 24 hours)
                if hours_ago < 24:
                    recency_factor = 0.8 - (hours_ago / 120)  # Decay from 0.8 to 0.6 over 24 hours
                else:
                    recency_factor = 0.6 - (min((hours_ago - 24) / 240, 0.4))  # Decay from 0.6 to 0.2 over 10 days
                    
                recency_factor = max(0.2, min(0.8, recency_factor))  # Constrain between 0.2 and 0.8
                
            except (ValueError, TypeError):
                pass
        
        heuristic_scores.append(recency_factor)
        
        # Calculate an average of all heuristic scores
        heuristic_importance = sum(heuristic_scores) / len(heuristic_scores)
        
        # Combine LLM assessment with heuristics
        if llm_importance is not None:
            # Weight LLM assessment more heavily (70%)
            importance = (0.3 * heuristic_importance) + (0.7 * llm_importance)
        else:
            # Use heuristics combined with base importance
            importance = (0.4 * base_importance) + (0.6 * heuristic_importance)
        
        # Apply category-based importance adjustment
        if metadata and 'category' in metadata:
            category = metadata['category']
            # Boost importance for certain categories
            if category in ['personal', 'security', 'credentials', 'confidential']:
                importance = min(1.0, importance * 1.2)
            # Reduce importance for certain categories
            elif category in ['general', 'greeting', 'small_talk']:
                importance = importance * 0.8
        
        # Apply final scaling and ensure the value is between 0 and 1
        return max(0.0, min(1.0, importance))

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

    async def _execute_neo4j_query_with_retry(self, query_func, *args, **kwargs):
        """
        Execute a Neo4j query with retry logic and rate limiting
        
        Args:
            query_func: Function that executes the Neo4j query
            *args: Arguments to pass to the query function
            **kwargs: Keyword arguments to pass to the query function
            
        Returns:
            Result of the query function
        """
        attempts = 0
        backoff_time = self.min_delay
        last_error = None
        
        while attempts < self.max_retries:
            try:
                # Check if we're approaching rate limits
                if self.neo4j_remaining_queries < 10 and time.time() < self.neo4j_reset_time:
                    wait_time = self.neo4j_reset_time - time.time() + 1
                    print(f"Neo4j rate limit approaching. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                
                # Execute the query
                with self.driver.session() as session:
                    result = query_func(session, *args, **kwargs)
                    
                # Update rate limit tracking (simple estimation)
                self.neo4j_remaining_queries = max(0, self.neo4j_remaining_queries - 1)
                
                return result
                
            except Exception as e:
                last_error = e
                attempts += 1
                
                # Check if this is a rate limiting error
                if "capacity" in str(e).lower() or "too many" in str(e).lower():
                    # Aggressive backoff for rate limiting
                    backoff_time = min(backoff_time * 2, self.max_delay)
                    self.neo4j_remaining_queries = 0
                    self.neo4j_reset_time = time.time() + 60  # Assume 1 minute reset
                else:
                    # Standard backoff for other errors
                    backoff_time = min(backoff_time * 1.5, self.max_delay)
                
                print(f"Neo4j query attempt {attempts} failed: {e}. Retrying in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
        
        # If we get here, all retry attempts failed
        print(f"All Neo4j query attempts failed after {self.max_retries} retries: {last_error}")
        raise last_error

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
