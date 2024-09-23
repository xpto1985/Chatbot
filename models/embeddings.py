from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

#Carrega o modelo de embeddings Hugging Face especificado.
def load_embeddings_model(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
    return HuggingFaceEmbeddings(model_name=model_name)

#Gera embeddings para uma consulta utilizando o modelo de embeddings fornecido.
def embed_query_with_model(query, embeddings_model):
    query_embedding = embeddings_model.embed_query(query)
    # Converter para array NumPy se necessário
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    return query_embedding
