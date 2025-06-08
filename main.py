from libraries import *  # Tüm kütüphaneleri al
from vector_store import load_vector_store

local_model_path = 'models/llama-2-7b-chat.Q2_K.gguf'

def load_llm():
    print("Llama-2 modeli yükleniyor...")
    llm = CTransformers(
        model=local_model_path,  
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.1,
            'repetition_penalty': 1.1,
            'context_length': 2048
        }
    )
    print("Llama-2 modeli yüklendi!")
    return llm


def create_prompt_template():
    """Prompt template oluştur"""
    template = """You are a helpful AI assistant. Try to answer the user's question as best as possible.
    If it's casual conversation, respond in casual conversational terms, like greetings.
    If the question is about LLM (Large Language Model), artificial intelligence, machine learning or related technical topics, use the provided context.
    If the question is a general conversation, a polite expression, a greeting, etc., answer naturally.
    Only say "I don't know enough about this" if you can't find information in context on technical topics.

    Bağlam: {context}

    Soru: {question}

    Cevap: """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )
    return prompt

def create_qa_chain():
    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store() # soruya uygun belgeleri bulmak için gerekli
    prompt = create_prompt_template()
    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2}) # kullanıcının sorusuna en yakın 2 metin parçasını bulur (embedding'e göre).
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff', # bütün parçaları tek bir 'context' içine koyup modele verir.
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt} 
    )
    return qa_chain


def generate_response(query, qa_chain):
    response = qa_chain.invoke({'query': query})
    return response['result'], response['source_documents']

def main():
    qa_chain = create_qa_chain() 
    print("Çıkmak için 'quit' yazın.")
    
    while True:
        query = input("\nSorunuz: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', 'çık']:
            print("Görüşürüz!")
            break
        
        if not query:
            continue
        
        answer, sources = generate_response(query, qa_chain)
        print(f"\nCevap: {answer}")
        # print("\nKaynaklar:")
        # for doc in sources:
        #     print(doc.metadata.get('source', 'No source'))

if __name__ == "__main__":
    main()