from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Cria um índice FAISS a partir de um ficheiro de texto e guarda-o localmente.
def create_faiss_index_from_text(file_path, db_path, embeddings):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)
    return db

#Cria um índice FAISS a partir de um ficheiro Excel, organizando os dados em documentos e guardando-os localmente.
def create_faiss_index_from_excel(file_path, db_path, embeddings, token_limit=1000):
    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_excel(file_path)
    tokenizer = AutoTokenizer.from_pretrained(embeddings.model_name)

    documents = []
    for _, row in df.iterrows():
        unit_name = row['Nome da Unidade Curricular']
        columns_to_include = { 
            # Define os pares chave-valor para o contexto
            "Professor(es)": row.get('Professor(es)', 'Professores não encontrados'),
            "Código": row.get('Código', 'Código não encontrado'),
            "ECTS": row.get('ECTS', 'ECTS não encontrados'),
            "Descrição": row.get('Descrição', 'Descrição não encontrada'),
            "Competências a Desenvolver": row.get('Competências a Desenvolver', 'Competências não encontradas'),
            "Temas": row.get('Temas', 'Temas não encontrados'),
            "Metodologia": row.get('Metodologia', 'Metodologia não encontrada'),
            "Bibliografia": row.get('Bibliografia Obrigatória', 'Bibliografia não encontrada'),
            "Recursos": row.get('Outros Recursos', 'Recursos não encontrados'),
            "Avaliação": row.get('Avaliação', 'Avaliação não encontrada'),
            "Plano de Trabalho": row.get('Plano de Trabalho', 'Plano de Trabalho não encontrado'),
            "Calendário Avaliação": row.get('Calendário Avaliação', 'Calendário não encontrado'),
        }

        current_context = f"Unidade Curricular: {unit_name}\n"
        current_tokens = tokenizer.tokenize(current_context)
        current_token_count = len(current_tokens)

        for column, content in columns_to_include.items():
            new_text = f"{column}: {content}\n"
            new_tokens = tokenizer.tokenize(new_text)
            new_token_count = len(new_tokens)

            if current_token_count + new_token_count > token_limit:
                documents.append(Document(page_content=current_context, metadata={"unit_name": unit_name}))
                current_context = f"Unidade Curricular: {unit_name}\n{new_text}"
                current_token_count = len(tokenizer.tokenize(current_context))
            else:
                current_context += new_text
                current_token_count += new_token_count

        if current_context.strip():
            documents.append(Document(page_content=current_context, metadata={"unit_name": unit_name}))
            
    # Cria o índice FAISS a partir dos documentos utilizando os embeddings fornecidos
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)
    return db

def load_faiss_index(db_path, embeddings):
    return FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
