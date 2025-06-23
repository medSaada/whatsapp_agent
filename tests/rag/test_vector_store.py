from app.services.rag import VectorStoreService, VectorStoreConfig
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
def test_vector_store_service():
    #create a config
    config = VectorStoreConfig(
        store_path="data/vector_store/test_store",
        collection_name="test_collection",
        embedding_model_name="openai"
    )
    #create a vector store service
    documents = [
    Document(
        page_content="L'intelligence artificielle est une technologie qui permet aux machines d'apprendre.",
        metadata={"source": "ai_guide.txt", "category": "technology"}
    ),
    Document(
        page_content="Le machine learning est un sous-ensemble de l'IA.",
        metadata={"source": "ml_guide.txt", "category": "technology"}
    ),
    Document(
        page_content="Les chatbots utilisent l'IA pour converser avec les utilisateurs.",
        metadata={"source": "chatbot_guide.txt", "category": "application"}
    )
]    
    #create a vector store service
    vector_service = VectorStoreService(config)
    #create a collection
    collection_name = "test_collection"
    vector_service.create_collection(collection_name, documents)
    #search a collection
    query = "What is the meaning of life?"
    results = vector_service.search_collection(collection_name, query)
    assert len(results) > 0
    #delete a collection
    vector_service.delete_collection(collection_name)
    # Nouveaux documents
 # Ajout de documents à une collection existante
    new_documents = [
        Document(
            page_content="Le deep learning utilise des réseaux de neurones profonds.",
            metadata={"source": "deep_learning.txt", "category": "technology"}
        )
    ]

    # Ajouter à la collection existante
    updated_info = vector_service.add_documents_to_collection(
        collection_name="faq",
        documents=new_documents
    )

    print(f"Documents ajoutés. Total: {updated_info.document_count}")

    #nettoyer
    vector_service.cleanup()
        
