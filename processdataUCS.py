import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import AutoTokenizer

DB_FAISS_PATH_EXCEL = 'vector_DB/db_faiss_excel'

def create_vector_db_from_excel(file_path, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', token_limit=1000):
    df = pd.read_excel(file_path)

    # Carregar o tokenizer do modelo Hugging Face para contar os tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Lista para armazenar os documentos criados
    documents = []

    # Iterar sobre cada linha do DataFrame (cada unidade curricular)
    for _, row in df.iterrows():
        unit_name = row['Nome da Unidade Curricular']

        # Criar pares chave-valor com as colunas e seus conteúdos
        columns_to_include = {
            "Professor(es)": row.get('Professor(es)', 'Professores não encontrados'),
            "Descrição": row.get('Descrição', 'Descrição não encontrada'),
            "Competências a Desenvolver": row.get('Competências a Desenvolver', 'Competências não encontradas'),
            "Temas": row.get('Temas', 'Temas não encontrados'),
            "Metodologia": row.get('Metodologia', 'Metodologia não encontrada'),
            "Bibliografia": row.get('Bibliografia Obrigatória', 'Bibliografia não encontrada'),
            "Recursos": row.get('Outros Recursos', 'Recursos não encontrados'),
            "Avaliação": row.get('Avaliação', 'Avaliação não encontrada'),
            "Plano de Trabalho": row.get('Plano de Trabalho', 'Plano de Trabalho não encontrado'),
            "Calendário Avaliação": row.get('Calendária Avaliação', 'Calendário não encontrado'),
        }

        current_context = f"Unidade Curricular: {unit_name}\n"
        current_tokens = tokenizer.tokenize(current_context)
        current_token_count = len(current_tokens)

        # Iterar sobre as colunas e adicionar ao contexto até o limite de 350 tokens
        for column, content in columns_to_include.items():
            # Criar o novo trecho a ser adicionado
            new_text = f"{column}: {content}\n"
            new_tokens = tokenizer.tokenize(new_text)
            new_token_count = len(new_tokens)

            # Verificar se adicionar esse trecho ultrapassa o limite de tokens
            if current_token_count + new_token_count > token_limit:
                # Se ultrapassar, mostrar informações do contexto
                print(f"Contexto finalizado para a Unidade Curricular '{unit_name}':")
                print(f"Tokens utilizados: {current_token_count}")
                print(current_context)
                print("-" * 50)

                # Criar o documento com o contexto atual
                documents.append(Document(page_content=current_context, metadata={"unit_name": unit_name}))

                # Resetar o contexto com o novo trecho
                current_context = f"Unidade Curricular: {unit_name}\n{new_text}"
                current_token_count = len(tokenizer.tokenize(current_context))
            else:
                # Adicionar o novo trecho ao contexto
                current_context += new_text
                current_token_count += new_token_count

        # Adicionar o último contexto restante, se houver
        if current_context.strip():
            print(f"Contexto finalizado para a Unidade Curricular '{unit_name}':")
            print(f"Tokens utilizados: {current_token_count}")
            print(current_context)
            print("-" * 50)

            documents.append(Document(page_content=current_context, metadata={"unit_name": unit_name}))

    # Criar embeddings com base nos documentos e salvar no FAISS
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(documents, embeddings)

    # Salvar o índice FAISS localmente
    db.save_local(DB_FAISS_PATH_EXCEL)

if __name__ == "__main__":
    # Executar a criação do banco de dados FAISS
    create_vector_db_from_excel('context/Dados_UCS.xlsx')
