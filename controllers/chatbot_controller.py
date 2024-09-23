from models.embeddings import load_embeddings_model
from models.faiss_index import load_faiss_index
from llama_cpp import Llama
from config.settings import MODEL_PATH, FAISS_PATHS
from controllers.database_controller import search_faiss_with_embed_query
import warnings
import os

# Ignora apenas warnings de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carrega o modelo de embeddings para gerar vetores semânticos
embeddings_model = load_embeddings_model()

# Inicializa o modelo LLaMA com o caminho especificado e contexto de 2500 tokens
llm = Llama(model_path=MODEL_PATH, n_ctx=2500, verbose=False)

# Guarda o histórico de perguntas e respostas num ficheiro de texto.
def save_history_to_file(history, file_path="data/historico/history.txt"):
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Abre o ficheiro em modo de append ("a"), criando-o se não existir
    with open(file_path, "a", encoding="utf-8") as file:
        for entry in history:
            file.write(f"Pergunta: {entry['pergunta']}\n")
            file.write(f"Resposta: {entry['resposta']}\n")
            file.write("-" * 50 + "\n")

# Processa a pergunta recebida, procura no contexto apropriado, e gera uma resposta
def handle_question(data):
    question = data.get('question')
    context = data.get('context')
    history = data.get('history', [])

    db_path = FAISS_PATHS.get(context)
    if not db_path:
        return {'error': 'Contexto não encontrado.'}

    db = load_faiss_index(db_path, embeddings_model)
    results = search_faiss_with_embed_query(db, question, embeddings_model)

    if results:
        context_text = "\n\n".join([doc.page_content for doc in results])
        response = generate_response(question, context_text, history)
        history.append({"pergunta": question, "resposta": response})

        # Guardar o histórico no ficheiro
        save_history_to_file(history)

        return {'answer': response, 'history': history}
    else:
        return {'answer': "Contexto relevante não encontrado.", 'history': history}

    
#Gera uma resposta baseada na pergunta, contexto fornecido, e histórico de interações.
def generate_response(question, context_text, history):
    messages = [
        {"role": "system", "content": "Você é um assistente chamado BART. Caso precises de mais dados para procurares a resposta pergunta ao utilizador. Responde estritamente com base no contexto fornecido e de forma curta e simpática. Caso a resposta não esteja no contexto, responda 'Não consigo responder a essa questão, experimente reformular a questão ou mudar de tema.' Não utilizes conhecimento prévio."},
        {"role": "system", "content": f"Contexto: {context_text}"}
    ]

    for pair in history[-3:]:
        messages.append({"role": "user", "content": pair['pergunta']})
        messages.append({"role": "assistant", "content": pair['resposta']})

    messages.append({"role": "user", "content": question})
    
    # Print do que vai para o modelo
    print("\n--- Prompt Enviado para o Modelo ---")
    print(f"Pergunta Atual: {question}\n")
    print(f"Contexto: {context_text}\n")
    print("Histórico (últimas 3 interações):")
    for pair in history[-3:]:
        print(f"Pergunta: {pair['pergunta']}")
        print(f"Resposta: {pair['resposta']}")
    print("\n--- Fim do Prompt ---\n")

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=250,
        temperature=0.2,
        top_p=0.9
    )

    return response['choices'][0]['message']['content'].strip()
