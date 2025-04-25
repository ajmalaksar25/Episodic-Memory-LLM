# Changelog

All notable changes to the Episodic Memory LLM project will be documented in this file.

## [1.3.2] - 2024-03-15

### Added
- Enhanced entity recognition with specialized detection for multi-word entities
- Extended pattern matching for technical entities, URLs, dates, and emails
- Improved entity extraction with prioritization for better recall
- Category-based diversity filtering for retrieved memories
- Adaptive query handling based on query complexity

### Improved
- Context preservation scoring algorithm with better semantic matching
- Entity recall with improved partial matching and normalization
- Memory scoring algorithm with better weights for context and relevance
- Memory retrieval with optimized search strategy selection
- Memory and entity caching for better performance

### Fixed
- Context preservation score decline (-4.78%) in previous version
- Relevance score decline (-16.44%) in previous version
- Entity recall decrease (-18.75%) in previous version
- Missing attribute in GroqProvider causing errors
- Memory caching with proper cache size management

## [1.3.1] - 2024-03-15

### Added
- Memory caching system for entity extraction to reduce processing time
- Response caching in the Groq provider to reduce API calls
- Conditional search methods that prioritize faster search techniques for simpler queries
- Context preservation factor in memory scoring to improve relevance
- Enhanced entity matching with partial match detection

### Improved
- Entity extraction process with faster algorithm and in-memory cache
- Memory recall efficiency with conditional searching based on query complexity
- LLM response generation with result caching
- Error handling throughout the Neo4j query system
- Entity recall with improved partial matching and normalization
- Context preservation scoring with semantic phrase matching

### Fixed
- Response time performance issues with the latest LLMs
- Low context preservation scores in complex queries
- Poor entity recall in responses
- Memory leaks due to growing context window
- Neo4j database query errors related to metadata properties

## [1.3.0] - 2023-12-05

### Added
- Enhanced entity extraction with improved NLP processing
- Multi-method memory recall combining vector, keyword, and entity-based search
- Adaptive scoring system for memory retrieval prioritization
- Category-based memory diversity to ensure varied context
- Semantic phrase matching for better context preservation
- Advanced entity recall with partial matching

### Improved
- Memory importance calculation with multi-factor scoring
- Context window management for more coherent conversations
- Entity extraction with regex patterns for technical terms
- Memory decay algorithm for better long-term memory management
- Conversation coherence calculation
- Handling of duplicate memories

### Fixed
- Entity recall issues in complex scenarios
- Context preservation scoring
- Relevance scoring for technical discussions
- Import issues in run_benchmarks.py
- Memory leaks in context window management

## [1.2.0] - 2023-09-15

### Added
- Basic entity extraction and relationship tracking
- Neo4j graph database integration for entity relationships
- Initial version of memory importance scoring
- Simple memory decay mechanism
- Basic context window management

### Improved
- Memory retrieval with vector search
- Query handling for follow-up questions
- Response generation using retrieved memories

### Fixed
- Memory duplication issues
- Missing imports in main modules
- Configuration loading errors 