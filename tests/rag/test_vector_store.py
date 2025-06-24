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
    vector_service = VectorStoreService(config, embedding_model=OpenAIEmbeddings())
    #create a collection
    collection_name = "test_collection"
    vector_service.create_collection(collection_name, documents)
    #load the collection
    collection = vector_service.load_collection(collection_name)
    assert collection is not None, "Collection should be loaded successfully"
    #search the collection
    query = "What is the main idea of the document?"
    results = vector_service.search_collection(collection_name, query)
    assert results.total_results > 0, "Search results should be returned successfully"
    # clean up before deleting the collection
    vector_service.cleanup()
    #delete the collection
    vector_service.delete_collection(collection_name)
    #nettoyer
 
        

