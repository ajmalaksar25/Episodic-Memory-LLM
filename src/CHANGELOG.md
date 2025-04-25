# Changelog

## [1.3.0] - 2023-12-05

### Added
- Enhanced entity extraction with improved pattern matching
- Multi-method memory recall combining vector, keyword, and entity-based approaches
- Adaptive scoring for memory prioritization based on multiple factors
- Category-based memory diversity in context window
- Semantic phrase matching for better context preservation
- Advanced entity recall with partial and alternative form matching

### Improved
- Memory importance calculation with multiple heuristics
- Context window management with better prioritization
- Entity extraction with regex patterns for emails, URLs, and dates
- Memory decay algorithm with reference count bonuses
- Conversation coherence calculation for better benchmarking
- Handling of duplicate memories with importance boosting

### Fixed
- Poor entity recall in episodic memory benchmarks
- Low context preservation scores compared to baseline
- Inconsistent relevance scoring in memory retrieval
- Import issues in run_benchmarks.py for better compatibility
- Memory leaks in context window management

## [1.2.1] - 2023-11-20

### Fixed
- Added missing _generate_accuracy_comparison method to BenchmarkingSystem
- Improved rate limiting protection in GroqProvider with exponential backoff
- Added retry mechanism for API calls to handle 429 errors
- Fixed resource cleanup in benchmarking to prevent memory leaks
- Added fallback index.html generation when visualizations fail
- Improved error handling throughout the benchmarking process

### Added
- Command-line options for controlling delays between model benchmarking
- Better progress reporting during benchmarking
- Configurable retry attempts for failed benchmarks

## [1.2.0] - 2023-11-15

### Added
- Enhanced benchmarking system with multi-model support
- Added detailed text output for benchmark results
- Implemented combined visualizations for model comparison
- Support for benchmarking on 6 different LLM models
- New command-line arguments for benchmarking flexibility
- Comprehensive documentation for the benchmarking system

### Changed
- Improved imports in run_benchmarks.py for better module compatibility
- Enhanced visualization quality and readability
- Better error handling and progress feedback

### Fixed
- Fixed inconsistent function naming for test scenarios
- Resolved issues with relative vs absolute imports
- Fixed path handling for visualization outputs

## [1.1.0] - 2023-10-20

### Added
- Initial benchmarking system implementation
- Basic visualization for single-model benchmarks
- Test scenarios for evaluating model performance

## [1.0.0] - 2023-09-15

### Added
- Initial release of Episodic Memory LLM
- Core memory management functionality
- LLM provider integrations
- Memory decay and cleanup mechanisms
- Entity extraction and relationship mapping 