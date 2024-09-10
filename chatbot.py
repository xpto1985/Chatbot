import os
from llama_cpp import Llama
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

# Importar as funções para criar as bases de dados FAISS
from processdataUCS import create_vector_db_from_excel
from processdataUAB import create_vector_db_from_text as create_vector_db_uab
from processdataMPV import create_vector_db_from_text as create_vector_db_mpv
from query import search_faiss_with_embed_query

# Ignora apenas warnings de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

def save_history_to_file(history):
    """Salva o histórico de perguntas e respostas em um arquivo de texto."""
    with open('history.txt', 'a', encoding='utf-8') as file:
        for entry in history:
            file.write(f"Pergunta: {entry['pergunta']}\n")
            file.write(f"Resposta: {entry['resposta']}\n\n")

# Chatbot
def chatbot(db):
    print("Bem-vindo ao Chatbot da UAB! Pergunte algo sobre o tema escolhido (digite 'sair' para encerrar):")

    # Lista para armazenar pares de perguntas e respostas
    conversation_history = []

    while True:
        instruction = input("\nFaça a sua pergunta: ")

        if instruction.lower() == "sair":
            print("Encerrando o Chatbot. Até logo!")
            break

        # Pesquisar o contexto no FAISS utilizando a query fornecida
        try:
            context = search_faiss_with_embed_query(db, instruction, embeddings_model)
            start_time = time.time()  # Capturar o tempo inicial
        except Exception as e:
            print(f"Erro ao procurar no FAISS: {e}")
            continue

        if context:
            # Formatar o contexto para exibição no prompt
            context_text = "\n\n".join([doc.page_content for doc in context])

            # Prepara as mensagens para o modelo incluindo o histórico recente
            messages = [
                {"role": "system", "content": "Você é um assistente chamado BART. Responde estritamente com base no contexto fornecido e de forma curta e simpática. Caso a resposta não esteja no contexto, responda 'Não consigo responder a essa questão, experimente reformular a questão ou mudar de tema.'Não utilizes conhecimento prévio, responde apenas sobre a pergunta e contexto."},
                {"role": "system", "content": f"Contexto: {context_text}"}
            ]

            # Adicionar as últimas 3 interações do histórico, se existirem
            if conversation_history:
                last_interactions = conversation_history[-3:]  # últimas 5 interações
                for pair in last_interactions:
                    messages.append({"role": "user", "content": pair['pergunta']})
                    messages.append({"role": "assistant", "content": pair['resposta']})

            # Adiciona a nova pergunta ao final das mensagens
            messages.append({"role": "user", "content": instruction})

            print("Mensagens enviadas:", messages) # Apenas para debug, pode ser removido
            # Gerar a resposta do modelo
            try:
                response = llm.create_chat_completion(
                    messages=messages,  # Envia o histórico recente, o contexto e a nova pergunta
                    max_tokens=250,      # Limite para a resposta
                    temperature=0.2,     # Controla a criatividade do modelo
                    top_p=0.9            # Filtro para tokens gerados
                )

                response_text = response['choices'][0]['message']['content'].strip()

                # Adicionar a nova pergunta e resposta ao histórico
                conversation_history.append({"pergunta": instruction, "resposta": response_text})

                # Salvar o histórico no arquivo de texto
                save_history_to_file(conversation_history)

                # Exibir a resposta gerada pelo modelo
                end_time = time.time()  # Capturar o tempo final
                elapsed_time = end_time - start_time  # Calcular o tempo de execução
                print(f"Tempo de resposta: {elapsed_time:.2f} segundos")  # Exibir o tempo de resposta

                print(f"\nResposta: {response_text}")
            except Exception as e:
                print("Erro ao gerar resposta:", e)
        else:
            print("\nContexto não encontrado na base de dados FAISS. Tente outra pergunta.")

        # Opcional: Exibir o histórico completo
        print("\nHistórico de Perguntas e Respostas (últimos 3):")
        for item in conversation_history[-3:]:  # Mostra apenas as últimas 5 interações
            print(f"Pergunta: {item['pergunta']}\nResposta: {item['resposta']}\n")


# Escolher qual base de dados FAISS utilizar
def choose_faiss_index():
    print("Escolha o contexto que deseja utilizar:")
    
    print("1. Aspetos Administrativos da UAB ")
    print("2. Conteúdos Programáticos das UCS de 3º ano, 2º Semestre.")
    print("3. Modelo Pedagógico Virtual")

    

    while True:
        choice = input("Escolha uma opção (1, 2 ou 3): ")
        
        if choice == '1':
            db_path = 'vector_DB/db_faiss_uab'
            return db_path
        elif choice == '2':
            db_path = 'vector_DB/db_faiss_excel'
            return db_path
        elif choice == '3':
            db_path = 'vector_DB/db_faiss_mpv'
            return db_path
        else:
            print("Escolha inválida. Por favor insira um valor entre 1 e 3.")

# Executar o chatbot

if __name__ == "__main__":

    # Só após o download, definir os caminhos e carregar os modelos
    model_path = "models/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, n_ctx=2500, verbose=False)

    verify_and_create_db()  # Verifica e cria as bases de dados FAISS se necessário
    db_path = choose_faiss_index()

    if db_path:
        # Carregar o modelo de embeddings após o download
        embeddings_model = load_embeddings_model('models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2')
        # Carregar a base de dados FAISS
        db = FAISS.load_local(db_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        # Iniciar o chatbot com a base de dados
        chatbot(db)
