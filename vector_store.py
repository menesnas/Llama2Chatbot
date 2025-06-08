from libraries import *
from collect_data import load_and_process_documents

def load_vector_store():
    print("Vector store yükleniyor...")
    # CPU kullan (daha stabil)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Vector store başarıyla yüklendi!")
        return db
    except Exception as e:
        print(f"Vector store yüklenemedi: {e}")
        print("Yeni vector store oluşturuluyor...")
        return load_and_process_documents()