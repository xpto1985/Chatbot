import os
from llama_cpp import Llama
from downloadmodels import download_gguf_model_if_needed, download_sentence_transformer_model_if_needed
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Importar as funções para criar os bases de dados FAISS
from processdataUCS import create_vector_db_from_excel
from processdataUAB import create_vector_db_from_text as create_vector_db_uab
from processdataMPV import create_vector_db_from_text as create_vector_db_mpv
from query import search_faiss_with_embed_query

# Ignora apenas warnings de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ajuste para evitar erro de OpenMP
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Função para carregar o modelo de embeddings
def load_embeddings_model(model_name='models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2'):
    return HuggingFaceEmbeddings(model_name=model_name)

# Verificar e criar diretórios se necessário
def verify_and_create_db():
    if not os.path.exists('vector_DB'):
        os.makedirs('vector_DB')

    # Verificar se cada base de dados FAISS existe e criar se necessário
    if not os.path.exists('vector_DB/db_faiss_excel'):
        print("base de dados db_faiss_excel não encontrado. Criando...")
        create_vector_db_from_excel('context/Dados_UCS.xlsx')
    
    if not os.path.exists('vector_DB/db_faiss_uab'):
        print("base de dados db_faiss_uab não encontrado. Criando...")
        create_vector_db_uab('context/UAB.txt')
    
    if not os.path.exists('vector_DB/db_faiss_mpv'):
        print("base de dados db_faiss_mpv não encontrado. Criando...")
        create_vector_db_mpv('context/MPV.txt')

# Chatbot
def chatbot(db):
    print("Bem-vindo ao Chatbot da UAB! Pergunte algo sobre o tema escolhido (digite 'sair' para encerrar):")

    while True:
        instruction = input("\nFaça sua pergunta: ")

        if instruction.lower() == "sair":
            print("Encerrando o Chatbot. Até logo!")
            break

        # Pesquisar o contexto no FAISS utilizando o a query fornecida
        try:
            context = search_faiss_with_embed_query(db, instruction, embeddings_model)
        except Exception as e:
            print(f"Erro ao procurar no FAISS: {e}")
            continue

        if context:
            # Formatar o contexto para exibição no prompt
            context_text = "\n\n".join([doc.page_content for doc in context])  # Combine o conteúdo dos documentos
            # Gerar o prompt completo no formato esperado pela API
            messages = [
                {"role": "system", "content": "Você é um assistente chamado BART. Responde estritamente com base no contexto fornecido e de forma curta e simpática. Caso a resposta não esteja no contexto, responda 'Não consigo responder a essa questão, experimente reformular a questão ou mudar de tema.'"},
                {"role": "system", "content": f"Contexto: {context_text}"},
                {"role": "user", "content": instruction}
            ]

            print("Mensagens enviadas:", messages) # Apenas para debug, pode ser removido

            # Gerar a resposta do modelo
            try:
                response = llm.create_chat_completion(
                    messages=messages,  # Mensagens no formato de chat
                    max_tokens=250,      # Limite para a resposta
                    temperature=0.2,     # Controla a criatividade do modelo
                    top_p=0.9            # Filtro para tokens gerados
                )

                # Exibir a resposta gerada pelo modelo
                print(f"\nResposta: {response['choices'][0]['message']['content'].strip()}")
            except Exception as e:
                print("Erro ao gerar resposta:", e)
        else:
            print("\nContexto não encontrado na base de dados FAISS. Tente outra pergunta.")

# Escolher qual base de dados FAISS utilizar
def choose_faiss_index():
    print("Escolha o contexto que deseja utilizar:")
    print("1. Conteúdos Programáticos das UCS de 3º ano, 2º Semestre.")
    print("2. Modelo Pedagógico Virtual")
    print("3. Aspetos Administrativos da UAB")

    choice = input("Digite o número correspondente ao contexto: ")

    if choice == '1':
        db_path = 'vector_DB/db_faiss_excel'
    elif choice == '2':
        db_path = 'vector_DB/db_faiss_mpv'
    elif choice == '3':
        db_path = 'vector_DB/db_faiss_uab'
    else:
        print("Escolha inválida. Encerrando.")
        return None

    return db_path

# Executar o chatbot

if __name__ == "__main__":

    # Só após o download, definir os caminhos e carregar os modelos
    model_path = "models/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    verify_and_create_db()  # Verifica e cria os bases de dados FAISS se necessário
    db_path = choose_faiss_index()

    if db_path:
        # Carregar o modelo de embeddings após o download
        embeddings_model = load_embeddings_model('models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2')
        # Carregar a base de dados FAISS
        db = FAISS.load_local(db_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        # Iniciar o chatbot com a base de dados carregada
        chatbot(db)
