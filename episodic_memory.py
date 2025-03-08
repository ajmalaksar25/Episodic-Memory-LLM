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
        """Initialize Neo4j schema with constraints"""
        with self.driver.session() as session:
            try:
                # Create indices for better performance
                session.run("""
                    CREATE INDEX memory_text IF NOT EXISTS 
                    FOR (m:Memory) ON (m.text)
                """)
                
                session.run("""
                    CREATE INDEX memory_timestamp IF NOT EXISTS 
                    FOR (m:Memory) ON (m.timestamp)
                """)

                # Create constraints
                session.run("""
                    CREATE CONSTRAINT memory_id IF NOT EXISTS 
                    FOR (m:Memory) REQUIRE m.id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT entity_name IF NOT EXISTS 
                    FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """)
            except Exception as e:
                print(f"Error initializing schema: {e}")
                raise
                   
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

    def _create_memory_with_entities(self, tx, memory_id: str, text: str, metadata: Dict, entities: Dict):
        """Create memory node and entity relationships in a single transaction"""
        try:
            # Create memory node
            memory_query = """
            CREATE (m:Memory {
                id: $memory_id,
                text: $text,
                timestamp: datetime(),
                category: $category,
                importance: $importance,
                sentiment: $sentiment,
                priority: $priority,
                conversation_id: $conversation_id,
                references: 0
            })
            RETURN m
            """
            
            # Ensure all required parameters are present
            memory_params = {
                "memory_id": memory_id,
                "text": text,
                "category": metadata.get("category", "general"),
                "importance": metadata.get("importance", 0.5),
                "sentiment": metadata.get("sentiment", "neutral"),
                "priority": metadata.get("priority", "MEDIUM"),
                "conversation_id": metadata.get("conversation_id", "default")
            }
            
            tx.run(memory_query, **memory_params)
            
            # Create or merge entities and relationships
            if entities:
                for entity_type, entity_list in entities.items():
                    for entity_name in entity_list:
                        # Create or merge entity node
                        entity_query = """
                        MERGE (e:Entity {name: $entity_name})
                        ON CREATE SET e.type = $entity_type,
                                    e.created_at = datetime()
                        WITH e
                        MATCH (m:Memory {id: $memory_id})
                        MERGE (m)-[r:CONTAINS]->(e)
                        ON CREATE SET r.created_at = datetime(),
                                    r.context_count = 1
                        ON MATCH SET r.context_count = r.context_count + 1,
                                    r.last_updated = datetime()
                        """
                        
                        tx.run(entity_query, {
                            "entity_name": entity_name.lower(),  # Normalize entity names
                            "entity_type": entity_type,
                            "memory_id": memory_id
                        })
                        
        except Exception as e:
            print(f"Error creating memory with entities: {e}")
            raise

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
        """Neo4j transaction for getting related memories with text similarity"""
        conditions = []
        if where_clause.get("category"):
            conditions.append("m.category = $category")
        if where_clause.get("conversation_id"):
            conditions.append("m.conversation_id = $conversation_id")
        if where_clause.get("importance", {}).get("$gte"):
            conditions.append("m.importance >= $min_importance")
            
        where_str = " AND ".join(conditions) if conditions else "TRUE"
        
        # Simple text matching without APOC
        search_condition = """
        AND (
            CASE
                WHEN $query <> ''
                THEN (
                    toLower(m.text) CONTAINS toLower($query)
                    OR any(word IN split(toLower($query), ' ')
                        WHERE toLower(m.text) CONTAINS word)
                )
                ELSE true
            END
        )
        """
        
        query_str = f"""
        MATCH (m:Memory)
        WHERE {where_str} {search_condition}
        WITH m,
             CASE
                 WHEN $query <> ''
                 THEN size([word IN split(toLower($query), ' ')
                           WHERE toLower(m.text) CONTAINS word]) * 1.0 /
                      size(split($query, ' '))
                 ELSE 1.0
             END as relevance_score
        ORDER BY relevance_score DESC, m.importance DESC, m.timestamp DESC
        LIMIT $top_k
        RETURN m.id as id,
               m.text as text,
               m.importance as importance,
               m.category as category,
               m.timestamp as timestamp,
               relevance_score
        """
        
        params = {
            "query": query.lower() if query else "",
            "category": where_clause.get("category"),
            "conversation_id": where_clause.get("conversation_id"),
            "min_importance": where_clause.get("importance", {}).get("$gte", 0),
            "top_k": top_k
        }
        
        try:
            result = tx.run(query_str, params)
            memories = []
            seen_texts = set()  # To prevent duplicates
            
            for record in result:
                memory_dict = dict(record)
                # Only add if we haven't seen this text before
                if memory_dict['text'] not in seen_texts:
                    seen_texts.add(memory_dict['text'])
                    # Round the relevance score for better readability
                    if 'relevance_score' in memory_dict:
                        memory_dict['relevance'] = round(memory_dict['relevance_score'], 3)
                    memories.append(memory_dict)
                    
            return memories
            
        except Exception as e:
            print(f"Error in memory recall: {e}")
            return []

    async def _check_memory_decay(self):
        """Check and apply memory decay using Neo4j"""
        try:
            current_time = datetime.now()
            decay_threshold = current_time - timedelta(hours=self.config.decay_interval)
            
            with self.driver.session() as session:
                session.execute_write(
                    self._apply_memory_decay_tx, 
                    decay_threshold.isoformat(),
                    self.config.importance_threshold,
                    self.config.min_references_to_keep
                )
        except Exception as e:
            print(f"Error in memory decay check: {e}")

    def _apply_memory_decay_tx(self, tx, decay_threshold: str, importance_threshold: float, min_references: int):
        """Apply memory decay in Neo4j"""
        query = """
        MATCH (m:Memory)
        WHERE m.last_accessed < datetime($decay_threshold)
        WITH m,
             m.importance * (1.0 - (duration.inDays(m.last_accessed, datetime($decay_threshold)).days * 0.1)) as decayed_importance
        WHERE decayed_importance < $importance_threshold
              AND coalesce(m.references, 0) < $min_references
        DETACH DELETE m
        """
        tx.run(query,
               decay_threshold=decay_threshold,
               importance_threshold=importance_threshold,
               min_references=min_references)

    def _get_conversation_memories_tx(self, tx, conversation_id: str) -> List[Dict]:
        """Get conversation memories from Neo4j"""
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
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics including age distribution"""
        with self.driver.session() as session:
            try:
                stats_query = """
                MATCH (m:Memory)
                WITH m,
                    CASE 
                        WHEN m.timestamp IS NOT NULL 
                        THEN duration.between(m.timestamp, datetime()) 
                        ELSE duration.between(datetime(), datetime())
                    END as age
                WITH 
                    count(m) as memory_count,
                    collect(distinct m.category) as categories,
                    sum(CASE WHEN m.timestamp IS NOT NULL AND age.days < 1 THEN 1 ELSE 0 END) as today,
                    sum(CASE WHEN m.timestamp IS NOT NULL AND age.days >= 1 AND age.days < 7 THEN 1 ELSE 0 END) as this_week,
                    sum(CASE WHEN m.timestamp IS NOT NULL AND age.days >= 7 AND age.days < 30 THEN 1 ELSE 0 END) as this_month,
                    sum(CASE WHEN m.timestamp IS NOT NULL AND age.days >= 30 THEN 1 ELSE 0 END) as older,
                    avg(coalesce(m.importance, 0)) as avg_importance
                RETURN {
                    memory_count: memory_count,
                    categories: categories,
                    age_distribution: {
                        today: today,
                        this_week: this_week,
                        this_month: this_month,
                        older: older
                    },
                    avg_importance: avg_importance
                } as stats
                """
                
                result = session.run(stats_query)
                return result.single()["stats"]
                
            except Exception as e:
                print(f"Error getting memory stats: {e}")
                return {
                    "memory_count": 0,
                    "categories": [],
                    "age_distribution": {
                        "today": 0,
                        "this_week": 0,
                        "this_month": 0,
                        "older": 0
                    },
                    "avg_importance": 0
                }

    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using a combination of NLP techniques.
        
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
                prompt = f"""
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
                """
                
                try:
                    llm_response = await self.llm_provider.generate_text(prompt, max_tokens=300)
                    
                    # Extract JSON from response using more robust method
                    import re
                    import json
                    
                    # Try to find JSON object in the response
                    json_match = re.search(r'\{[\s\S]*\}', llm_response)
                    if json_match:
                        json_str = json_match.group(0)
                        # Clean up common JSON formatting issues
                        json_str = re.sub(r',\s*\}', '}', json_str)  # Remove trailing commas
                        json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas in arrays
                        
                        try:
                            llm_entities = json.loads(json_str)
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
                        except json.JSONDecodeError as je:
                            print(f"Error parsing JSON from LLM response: {je}")
                except Exception as e:
                    print(f"Error extracting entities with LLM: {e}")
            
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

async def main():
    try:
        # Initialize the system with GroqProvider
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        model_name = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
        groq_provider = GroqProvider(api_key=api_key, model_name=model_name)
        
        memory_module = EpisodicMemoryModule(
            llm_provider=groq_provider,
            collection_name="episodic_memory",
            embedding_model="all-MiniLM-L6-v2",
        )
        
        print("\n=== Episodic Memory LLM Demo ===")

        # Test Case 1: Store diverse memories with LLM analysis
        print("\n1. Storing Test Memories...")
        test_memories = [
            {
                "text": "User implemented a neural network using PyTorch's autograd for gradient computation",
                "category": "programming",
                "priority": MemoryPriority.HIGH
            },
            {
                "text": "Discussed the mathematics behind backpropagation and chain rule in neural networks",
                "category": "machine_learning",
                "priority": MemoryPriority.HIGH
            },
            {
                "text": "Explored different optimization algorithms like Adam, SGD, and RMSprop",
                "category": "machine_learning",
                "priority": MemoryPriority.MEDIUM
            }
        ]

        conversation_id = "learning_session_001"
        for memory in test_memories:
            # Use LLM to analyze text before storing
            analysis = await groq_provider.analyze_text(memory["text"])
            
            # Merge LLM analysis with provided metadata
            metadata = {
                "category": memory["category"],
                "importance": analysis.get("importance", 0.5),
                "sentiment": analysis.get("sentiment", "neutral"),
                "summary": analysis.get("summary", ""),
            }
            
            memory_id = await memory_module.store_memory(
                text=memory["text"],
                metadata=metadata,
                priority=memory["priority"],
                conversation_id=conversation_id
            )
            if memory_id:
                print(f"Successfully stored memory: {memory_id}")
                print(f"Analysis: {json.dumps(analysis, indent=2)}")

        # Test Case 2: Memory Recall with LLM-enhanced summaries
        print("\n2. Testing Memory Recall...")
        recall_tests = [
            {
                "query": "neural network implementation",
                "category": "programming",
                "description": "Programming-specific recall"
            },
            {
                "query": "optimization algorithms",
                "category": "machine_learning",
                "description": "Machine learning concepts"
            },
            {
                "query": "PyTorch usage",
                "category": None,
                "description": "General technical query"
            }
        ]

        for test in recall_tests:
            print(f"\nRecall Test: {test['description']}")
            memories = await memory_module.recall_memories(
                query=test["query"],
                category=test["category"],
                conversation_id=conversation_id,
                top_k=2
            )
            
            if memories:
                print(f"Found {len(memories)} memories:")
                for mem in memories:
                    print(f"- {mem['text']}")
                    print(f"  Importance: {mem.get('importance', 'N/A')}")
                    print(f"  Relevance: {mem.get('relevance', 'N/A')}")
                
                # Generate a summary of recalled memories
                summary = await groq_provider.summarize_memories(
                    memories=memories,
                    query=test["query"]
                )
                print("\nMemory Summary:")
                print(summary)

        # Test Case 3: Entity Relationship Analysis
        print("\n5. Entity Relationships Analysis:")
        relationships = memory_module.get_entity_relationships()
        if relationships:
            print("\nFound Entity Relationships:")
            for rel in relationships[:3]:  # Show top 3 relationships
                print(f"\nAnalyzing relationship between '{rel['entity1']}' and '{rel['entity2']}':")
                
                # Use LLM to analyze relationship
                analysis = await groq_provider.analyze_relationships(
                    entity1=rel['entity1'],
                    entity2=rel['entity2'],
                    contexts=rel['shared_contexts']
                )
                
                print(f"Type: {analysis.get('relationship_type', 'unknown')}")
                print(f"Strength: {analysis.get('strength', 0):.2f}")
                print(f"Description: {analysis.get('description', '')}")
                if analysis.get('key_interactions'):
                    print("Key Interactions:")
                    for interaction in analysis['key_interactions']:
                        print(f"- {interaction}")
        else:
            print("No significant entity relationships found")

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
