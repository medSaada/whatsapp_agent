#  WhatsApp Customer Service AI Agent for Moroccan Market

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-purple.svg)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent **multilingual WhatsApp customer service AI agent** designed specifically for **Moroccan customers**. Built with **LangGraph orchestration** and **RAG (Retrieval Augmented Generation)**, this agent seamlessly handles conversations in **Darija (Moroccan Arabic)**, **French**, and **English**, providing contextually accurate responses for **Geniats** e-learning academy.

## 🎯 Project Overview

This project creates an AI-powered customer service agent that:
- **Speaks Like a Local**: Native-level conversations in Darija, French, and English
- **Understands Context**: Uses RAG to retrieve relevant information from knowledge base
- **Orchestrates Intelligently**: LangGraph manages complex conversation flows and decision-making
- **Serves Moroccan Market**: Culturally adapted responses for Moroccan customers

## 🌟 Core Features

### 🗣️ **Multilingual Communication**
- **Darija (Moroccan Arabic)**: Natural conversations using Arabic script with proper punctuation (،؟!)
- **French**: Professional business communication for formal inquiries
- **English**: International customer support capabilities
- **Language Detection**: Automatically responds in the user's preferred language
- **Cultural Adaptation**: Moroccan-specific greetings, expressions, and business etiquette

### 🧠 **LangGraph-Powered Intelligence**
- **State Machine Orchestration**: LangGraph manages conversation flow and decision trees
- **Multi-Agent Architecture**: Planner → Tool Caller → Generator workflow
- **Conversation Memory**: Persistent memory with intelligent summarization
- **Context Switching**: Seamless transitions between different conversation topics
- **Tool Integration**: Dynamic tool selection based on user queries

### 📚 **RAG (Retrieval Augmented Generation)**
- **Vector Database**: Chroma/Qdrant for semantic document retrieval
- **Smart Chunking**: Intelligent document processing and indexing
- **Context-Aware Responses**: Retrieves relevant information before generating answers
- **Knowledge Base**: Comprehensive information about courses, pricing, and policies
- **Real-time Updates**: Dynamic knowledge base updates without retraining

### 🔧 **Technical Infrastructure**
- **FastAPI Backend**: High-performance async Python web framework
- **WhatsApp Business API**: Native Meta integration for message processing
- **Webhook Processing**: Real-time message handling and status tracking
- **OpenAI Integration**: GPT-4.1 for generation, embeddings for retrieval
- **Monitoring**: LangSmith integration for conversation analytics

## 🏗️ LangGraph Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WhatsApp      │    │   FastAPI        │    │   LangGraph     │
│   Business API  │◄──►│   Webhook        │◄──►│   Orchestrator  │
└─────────────────┘    │   Handler        │    └─────────────────┘
                       └──────────────────┘              │
                                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │              LangGraph State Machine            │
                    │                                                 │
                    │  ┌─────────┐   ┌─────────────┐   ┌───────────┐ │
                    │  │ Planner │──►│ Tool Caller │──►│ Generator │ │
                    │  │  Node   │   │    Node     │   │   Node    │ │
                    │  └─────────┘   └─────────────┘   └───────────┘ │
                    │                        │                       │
                    │                        ▼                       │
                    │              ┌─────────────────┐               │
                    │              │   RAG Tools     │               │
                    │              │ Vector Retrieval│               │
                    │              └─────────────────┘               │
                    └─────────────────────────────────────────────────┘
                                             │
                                             ▼
                                   ┌─────────────────┐
                                   │   Vector Store  │
                                   │ (Chroma/Qdrant) │
                                   └─────────────────┘
```

### 🗂️ Project Structure

```
whatsapp_agent_poc/
├── app/
│   ├── api/v1/              # FastAPI routes and webhook endpoints
│   ├── core/
│   │   ├── config.py        # Environment configuration
│   │   ├── prompt.py        # LangGraph node prompts (Planner, Generator)
│   │   └── logging.py       # Comprehensive logging system
│   ├── schemas/             # Pydantic models for WhatsApp payloads
│   └── services/
│       ├── whatsapp_service.py     # Main WhatsApp message orchestration
│       ├── meta_api_client.py      # Meta API integration
│       └── rag/                    # RAG System Components
│           ├── orchestrator.py     # LangGraph workflow coordinator
│           ├── generation_service.py  # LLM response generation
│           ├── vector_store_service.py # Document retrieval engine
│           ├── chunking_service.py    # Document processing pipeline
│           └── graph/              # LangGraph Implementation
│               ├── builder.py      # Graph construction and state management
│               ├── nodes.py        # Individual LangGraph nodes
│               ├── tools.py        # RAG tools for document retrieval
│               └── state.py        # Conversation state definitions
├── data/
│   ├── documents/           # Knowledge base (courses, pricing, FAQ)
│   ├── vector_store/        # Embedded document storage
│   └── sqlite/              # Conversation memory and user sessions
├── tests/                   # Test suite and LangGraph Studio integration
├── main.py                  # FastAPI application entry point
├── ingest.py               # Knowledge base ingestion pipeline
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with Conda environment management
- **WhatsApp Business API** access (Meta Developer Account)
- **OpenAI API** key for GPT-4.1 and embeddings
- **ngrok** for local development and webhook testing

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n app-whatsapp-agent python=3.8
conda activate app-whatsapp-agent

# Install all dependencies including LangGraph
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with required API keys:

```env
# WhatsApp Business API (Meta)
META_ACCESS_TOKEN=your_meta_access_token
META_VERIFY_TOKEN=your_verify_token
META_WABA_ID=your_whatsapp_business_account_id
META_PHONE_NUMBER_ID=your_phone_number_id

# OpenAI for LLM and Embeddings
OPENAI_API_KEY=your_openai_api_key

# Additional AI Services
HF_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_api_key

# LangSmith Monitoring (Optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=moroccan_whatsapp_agent
LANGCHAIN_TRACING_V2=true

# Knowledge Base Documents
DOCUMENT_PATH_1=data/documents/manual_data_fz.txt
DOCUMENT_PATH_2=data/documents/datagenerated_assistant.txt
```

### 3. Initialize Knowledge Base

```bash
# Process and embed documents into vector store
python ingest.py
```

### 4. Launch the Application

```bash
# Start FastAPI server with LangGraph orchestration
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Setup Development Webhook

```bash
# Expose local server for WhatsApp webhook testing
ngrok http 8000
```

Configure WhatsApp webhook URL: `https://your-ngrok-url.ngrok.io/api/v1/webhook`


## 🔧 LangGraph Configuration

### RAG Orchestrator Settings

```python
RAGOrchestrator(
    settings=settings,
    vector_store_path="data/vector_store",
    collection_name="production_collection",
    model_name="gpt-4.1",           # Primary LLM for generation
    temperature=0.2,                # Controlled creativity
    memory_threshold=6              # Conversation context limit
)
```

### LangGraph Node Customization

Edit `app/core/prompt.py` to modify:
- **Planner Prompt**: Decision-making logic for tool usage
- **Generator Prompt**: Response generation with cultural context
- **Summarizer Prompt**: Conversation memory management

### Multi-Language Support Configuration

The agent automatically detects and responds in:
1. **Darija**: Arabic script with Moroccan expressions
2. **French**: Formal business communication
3. **English**: International customer support

## 🧪 Testing & Development

```bash
# Run comprehensive test suite
pytest

# Test LangGraph workflows in isolation
python tests/studio_test.py

# Monitor conversations in LangSmith
# Access your project dashboard for real-time analytics
```

## 📊 LangGraph Monitoring

### Conversation Flow Tracking
- **Node Execution**: Monitor which LangGraph nodes are triggered
- **Tool Usage**: Track RAG retrieval effectiveness
- **State Transitions**: Analyze conversation flow patterns
- **Performance Metrics**: Response times and success rates

### LangSmith Integration
- **Real-time Debugging**: Step-by-step conversation analysis
- **A/B Testing**: Compare different prompt configurations
- **Error Analysis**: Identify and resolve conversation failures
- **User Analytics**: Understand customer interaction patterns

## 🌍 Moroccan Market Focus

### Cultural Adaptations
- **Greetings**: Proper Islamic greetings and responses
- **Business Etiquette**: Moroccan professional communication standards
- **Local Context**: Understanding of Moroccan education system and family dynamics
- **Pricing Display**: Moroccan Dirham (MAD) currency formatting

### Language Switching Logic
- **Auto-Detection**: Identifies user language from first message
- **Consistent Response**: Maintains same language throughout conversation
- **Mixed Language Handling**: Graceful handling of code-switching between languages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/langgraph-enhancement`)
3. Test your changes with LangGraph Studio
4. Commit with clear descriptions (`git commit -m 'Add multilingual node optimization'`)
5. Push and create a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for orchestration framework
- **LangChain** for RAG implementation
- **OpenAI** for multilingual language models
- **Meta** for WhatsApp Business API
- **Geniats Academy** for educational mission in Morocco

---

**🇲🇦 Built for Morocco - Empowering Moroccan families through AI-powered education support** 
