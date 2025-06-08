import os
import arxiv
from libraries import *

# ArXiv paper ID'leri - LLM konusunda önemli makaleler
ARXIV_PAPERS = [
    '2308.10620',  # LLaMA 2
    '2307.06435',  # Llama 2 Chat
    '2303.18223',  # GPT-4
    '2307.10700',  # Code Llama
    '2310.11207',  # Mistral 7B
    '2305.11828',  # Vicuna
    '2307.09288',  # Longformer
    '2305.15334',  # PaLM 2
    '1706.03762',  # Attention is all you need
]

def download_arxiv_papers():
    """ArXiv'den makaleleri indir"""
    
    data_dir = "./data"
    
    # Data klasörünü oluştur
    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 Data klasörü hazır: {data_dir}")
    
    downloaded_papers = []
    
    print(f"📚 {len(ARXIV_PAPERS)} ArXiv makalesi kontrol ediliyor...")
    
    for i, paper_id in enumerate(ARXIV_PAPERS, 1):
        filename = f"{paper_id}.pdf"
        file_path = os.path.join(data_dir, filename)
        
        # Dosya zaten varsa atla
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100000:  # En az 100KB
            print(f"✅ ({i}/{len(ARXIV_PAPERS)}) {filename} zaten mevcut")
            downloaded_papers.append(file_path)
            continue
        
        try:
            print(f"📥 ({i}/{len(ARXIV_PAPERS)}) {paper_id} indiriliyor...")
            
            # ArXiv'den makale al
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            # PDF'i indir
            paper.download_pdf(dirpath=data_dir, filename=filename)
            
            if os.path.exists(file_path) and os.path.getsize(file_path) > 100000:
                print(f"✅ Başarılı: {filename}")
                downloaded_papers.append(file_path)
            else:
                print(f"❌ Başarısız: {filename}")
                
        except Exception as e:
            print(f"❌ {paper_id} indirilemedi: {e}")
            continue
    
    print(f"📁 Toplam {len(downloaded_papers)} makale hazır")
    return downloaded_papers

def load_and_process_documents():
    """Belgeleri yükle ve işle"""
    
    # ArXiv makalelerini indir
    download_arxiv_papers()
    
    dirpath = './data'
    
    # PDF dosyalarını kontrol et
    pdf_files = [f for f in os.listdir(dirpath) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ Hiç PDF bulunamadı!")
        return None
    
    print(f"📁 {len(pdf_files)} PDF dosyası bulundu")
    
    # PDF dosyalarını yükle
    loader = DirectoryLoader(path=dirpath, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"📖 Toplam {len(documents)} sayfa yüklendi")
    
    if not documents:
        print("❌ Hiç belge yüklenemedi!")
        return None
    
    # Metinleri parçalara böl
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = splitter.split_documents(documents)
    print(f"📄 Toplam {len(texts)} metin parçası oluşturuldu")
    
    # GPU kullanılabilirliğini kontrol et
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Embedding için kullanılan cihaz: {device}")
    
    # Embedding modelini yükle
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    # Vector store veritabanını oluştur
    print("🔗 Vector store oluşturuluyor...")
    db = FAISS.from_documents(texts, embeddings)
    
    # Vector store'u kaydet
    db.save_local("faiss_index")
    print("💾 Vector store kaydedildi!")
    return db

if __name__ == "__main__":
    print("🚀 Sistem başlatılıyor...")
    db = load_and_process_documents()
    
    if db:
        print("✅ Sistem hazır!")
    else:
        print("❌ Sistem başlatılamadı!")