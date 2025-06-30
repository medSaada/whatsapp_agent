# Memory Management Feature

## Overview

The RAG agent now includes intelligent memory management that automatically summarizes and wipes conversation history when a threshold is reached. This prevents the conversation from becoming too long and helps maintain context while reducing memory usage.

## How It Works

### Configuration
- **Memory Threshold**: Set to 6 interactions by default (configurable)
- **Trigger**: When the interaction count reaches the threshold
- **Action**: Summarizes the conversation and wipes the message history
- **Result**: Starts fresh with the summary as context

### Logging

The system provides detailed logging for memory management events:

```
[Memory Management] Memory threshold set to: 6 interactions
[Memory Management] First interaction. Counter: 1
[Memory Management] Continuing conversation. Counter: 2/6
[Memory Management] Interaction count: 5 -> 6 (threshold: 6)
[Memory Management] 🧹 THRESHOLD REACHED! Summarizing and wiping conversation history...
[Memory Management] Messages to summarize: 12
[Memory Management] Found 10 conversation messages to summarize
[Memory Management] 📝 SUMMARY CREATED: The user asked about Geniats programs, pricing, and enrollment process...
[Memory Management] ✅ Memory wiped and reset. Starting fresh with summary.
```

### Key Features

1. **Automatic Summarization**: Uses the LLM to create concise summaries of conversations
2. **Threshold-Based Triggering**: Only activates when the interaction count reaches the configured threshold
3. **Context Preservation**: The summary is preserved as a system message for future context
4. **Robust Error Handling**: If summarization fails, the system continues without interruption
5. **Detailed Logging**: Comprehensive logs show when memory management events occur

### Configuration

To change the memory threshold, modify the `memory_threshold` parameter when initializing the RAGOrchestrator:

```python
orchestrator = RAGOrchestrator(
    vector_store_path="data/vector_store",
    collection_name="production_collection",
    model_name="gpt-4.1",
    temperature=0.2,
    memory_threshold=10  # Change to desired number of interactions
)
```

### Testing

Use the provided test script to see memory management in action:

```bash
python test_memory_management.py
```

This script simulates multiple interactions with a low threshold (3) to demonstrate the summarization and wiping process.

### Benefits

- **Prevents Memory Bloat**: Keeps conversation history manageable
- **Maintains Context**: Preserves important information through summaries
- **Improves Performance**: Reduces the amount of context sent to the LLM
- **Scalable**: Works automatically without manual intervention
- **Transparent**: Detailed logging shows exactly when and what happens 