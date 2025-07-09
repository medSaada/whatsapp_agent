# 🤖 WhatsApp Agent for Geniats E-Learning Academy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-purple.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated AI-powered WhatsApp customer service agent for **Geniats**, an e-learning coding academy for Moroccan children aged 6-15. This agent provides intelligent, multilingual customer support using advanced RAG (Retrieval Augmented Generation) capabilities.

## 🌟 Features

### 🎯 Core Capabilities
- **Intelligent Customer Service**: AI-powered responses using GPT-4 with custom persona
- **Multilingual Support**: Seamless conversation in Arabic (Darija) and French
- **RAG Integration**: Context-aware responses using vector store knowledge base
- **WhatsApp Business API**: Native integration with Meta's WhatsApp platform
- **Conversation Memory**: Persistent conversation tracking and summarization
- **Real-time Processing**: Immediate response to incoming messages

### 🧠 AI & Machine Learning
- **Advanced Language Models**: GPT-4.1 for generation, OpenAI embeddings for retrieval
- **Vector Database**: Chroma/Qdrant for efficient document retrieval
- **LangGraph Orchestration**: State machine for complex conversation flows
- **Semantic Chunking**: Intelligent document processing and indexing
- **Custom Personas**: "Fatima-Zahra" - culturally appropriate customer service agent

### 🔧 Technical Features
- **FastAPI Backend**: Modern, async Python web framework
- **Webhook Processing**: Real-time message handling from WhatsApp
- **Template Messages**: Automated responses for common queries
- **Status Tracking**: Message delivery and read receipt monitoring
- **Error Handling**: Robust logging and error management
- **LangSmith Integration**: Advanced monitoring and debugging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WhatsApp      │    │   FastAPI        │    │   RAG System    │
│   Business API  │◄──►│   Application    │◄──►│   (LangGraph)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Webhook        │    │   Vector Store  │
                       │   Processing     │    │   (Chroma/      │
                       └──────────────────┘    │   Qdrant)       │
                                               └─────────────────┘
```

### 🗂️ Project Structure

```
whatsapp_agent_poc/
├── app/
│   ├── api/v1/              # API routes and endpoints
│   ├── core/                # Configuration, logging, prompts
│   ├── schemas/             # Pydantic models for data validation
│   └── services/
│       ├── whatsapp_service.py    # WhatsApp message processing
│       ├── meta_api_client.py     # Meta API integration
│       └── rag/                   # RAG system components
│           ├── orchestrator.py    # Main RAG coordinator
│           ├── generation_service.py  # LLM response generation
│           ├── vector_store_service.py # Document retrieval
│           ├── chunking_service.py    # Document processing
│           └── graph/             # LangGraph state machine
├── data/
│   ├── documents/           # Knowledge base documents
│   ├── vector_store/        # Embedded document storage
│   └── sqlite/              # Conversation memory
├── tests/                   # Test suite
├── main.py                  # Application entry point
├── ingest.py               # Data ingestion script
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Conda (recommended for environment management)
- WhatsApp Business API access
- OpenAI API key
- ngrok (for local development)

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n app-whatsapp-agent python=3.8
conda activate app-whatsapp-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Meta WhatsApp Business API
META_ACCESS_TOKEN=your_meta_access_token
META_VERIFY_TOKEN=your_verify_token
META_WABA_ID=your_whatsapp_business_account_id
META_PHONE_NUMBER_ID=your_phone_number_id

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Additional AI Services
HF_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_api_key

# LangSmith (Optional - for monitoring)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true

# Document paths for knowledge base
DOCUMENT_PATH_1=data/documents/manual_data_fz.txt
DOCUMENT_PATH_2=data/documents/datagenerated_assistant.txt
```

### 3. Data Ingestion

Prepare the knowledge base by ingesting your documents:

```bash
python ingest.py
```

### 4. Run the Application

```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Setup Webhook (Development)

```bash
# In a separate terminal, expose your local server
ngrok http 8000
```

Configure your WhatsApp webhook URL: `https://your-ngrok-url.ngrok.io/api/v1/webhook`

## 📖 Usage

### WhatsApp Integration

The agent automatically processes incoming WhatsApp messages and provides intelligent responses based on:

1. **Predefined Knowledge Base**: Information about Geniats courses, pricing, and policies
2. **Conversation Context**: Maintains conversation history for coherent dialogue
3. **Cultural Adaptation**: Responds appropriately in Arabic (Darija) or French
4. **Business Logic**: Handles enrollment inquiries, course information, and support requests

### API Endpoints

- `GET /` - Health check
- `POST /api/v1/webhook` - WhatsApp webhook for message processing
- `GET /api/v1/webhook` - Webhook verification

### Example Conversation

**User (Arabic):** السلام عليكم، شنو العرض ديالكوم؟
**Agent:** وعليكم السلام! أنا فاطمة-زهراء من جنياتس. العرض ديالنا هو دورات برمجة بلغة Scratch للأطفال من 6 حتى 15 سنة...

## 🔧 Advanced Configuration

### RAG System Tuning

Modify `app/services/rag/orchestrator.py`:

```python
RAGOrchestrator(
    settings=settings,
    vector_store_path="data/vector_store",
    collection_name="production_collection",
    model_name="gpt-4.1",           # Change AI model
    temperature=0.2,                # Adjust creativity
    memory_threshold=6              # Conversation memory limit
)
```

### Custom Personas

Edit prompts in `app/core/prompt.py` to customize the agent's personality and responses.

### Vector Store Options

The system supports multiple vector stores:
- **Chroma** (default): Local, file-based vector storage
- **Qdrant**: Cloud-based vector database for production

## 🧪 Testing

```bash
# Run the test suite
pytest

# Test with LangGraph Studio
python tests/studio_test.py
```

## 📊 Monitoring

The application includes comprehensive logging and optional LangSmith integration for:

- **Request/Response Tracking**: Monitor all WhatsApp interactions
- **Performance Metrics**: Response times and success rates
- **Error Analysis**: Detailed error logs and debugging
- **Conversation Analytics**: User engagement and satisfaction metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **FastAPI** for the web framework
- **OpenAI** for language models
- **Meta** for WhatsApp Business API
- **Geniats Academy** for the educational mission

## 📞 Support

For support and questions:
- Create an issue in this repository
- Contact the development team
- Check the [documentation](docs/) for detailed guides

---

**Built with ❤️ for Geniats - Empowering the next generation of Moroccan programmers** 