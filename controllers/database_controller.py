import os
from models.faiss_index import create_faiss_index_from_text, create_faiss_index_from_excel
from models.embeddings import load_embeddings_model
from config.settings import FAISS_PATHS, CONTEXT_FILES
import numpy as np
import warnings

# Ignora apenas warnings de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carrega o modelo de embeddings para geração de vetores semânticos
embeddings_model = load_embeddings_model()

#Verifica se os diretórios e índices FAISS existem; caso contrário, cria os índices necessários.
def verify_and_create_db():
    if not os.path.exists('vector_DB'):
        os.makedirs('vector_DB')

    if not os.path.exists(FAISS_PATHS['1']):
        print("Criando o índice FAISS para UAB...")
        create_faiss_index_from_text(CONTEXT_FILES['1'], FAISS_PATHS['1'], embeddings_model)

    if not os.path.exists(FAISS_PATHS['2']):
        print("Criando o índice FAISS para MPV...")
        create_faiss_index_from_text(CONTEXT_FILES['2'], FAISS_PATHS['2'], embeddings_model)

    if not os.path.exists(FAISS_PATHS['3']):
        print("Criando o índice FAISS para UCS Excel...")
        create_faiss_index_from_excel(CONTEXT_FILES['3'], FAISS_PATHS['3'], embeddings_model)

#Realiza uma pesquisa num índice FAISS usando uma consulta embutida.
def search_faiss_with_embed_query(db, query, embeddings_model, top_k=2):
    # Gerar embeddings da consulta usando embed_query
    query_embedding = embeddings_model.embed_query(query)

    # Converter o embedding para um array NumPy se for uma lista
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)

    # Realizar a pesquisa no índice FAISS
    D, I = db.index.search(query_embedding.reshape(1, -1), top_k)

    # Recuperar os contextos correspondentes aos resultados mais relevantes
    results = []
    for i in I[0]:
        doc_id = db.index_to_docstore_id.get(i)  
        if doc_id in db.docstore._dict:
            document = db.docstore._dict[doc_id]
            results.append(document)
        else:
            print(f"Documento com ID {doc_id} não encontrado no docstore.")

    return results if results else None  # Retorna a lista de documentos encontrados ou None se estiver vazia