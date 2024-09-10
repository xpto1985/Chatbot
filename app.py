from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import os
import warnings
from processdataUCS import create_vector_db_from_excel
from processdataUAB import create_vector_db_from_text as create_vector_db_uab
from processdataMPV import create_vector_db_from_text as create_vector_db_mpv
from query import search_faiss_with_embed_query
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__, static_folder='web', static_url_path='/web')
CORS(app)

def load_embeddings_model(model_name='models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2'):
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings_model = load_embeddings_model()

def verify_and_create_db():
    if not os.path.exists('vector_DB'):
        os.makedirs('vector_DB')

    if not os.path.exists('vector_DB/db_faiss_excel'):
        print("Base de dados db_faiss_excel não encontrada. Criando...")
        create_vector_db_from_excel('context/Dados_UCS.xlsx')

    if not os.path.exists('vector_DB/db_faiss_uab'):
        print("Base de dados db_faiss_uab não encontrada. Criando...")
        create_vector_db_uab('context/UAB.txt')

    if not os.path.exists('vector_DB/db_faiss_mpv'):
        print("Base de dados db_faiss_mpv não encontrada. Criando...")
        create_vector_db_mpv('context/MPV.txt')

model_path = "models/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=2500, verbose=False)

def save_history_to_file(history):
    """Salva o histórico de perguntas e respostas num arquivo de texto."""
    with open('history.txt', 'a', encoding='utf-8') as file:
        for entry in history:
            file.write(f"Pergunta: {entry['pergunta']}\n")
            file.write(f"Resposta: {entry['resposta']}\n\n")

def load_faiss_index(context):
    if context == '1':
        db_path = 'vector_DB/db_faiss_uab'
    elif context == '2':
        db_path = 'vector_DB/db_faiss_mpv'
    elif context == '3':
        db_path = 'vector_DB/db_faiss_excel'
    else:
        return None

    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    else:
        return None

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/web/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    conversation_history = data.get('history', [])  # Recebe o histórico de conversas do frontend

    if not question or not context:
        return jsonify({'error': 'Pergunta ou contexto não fornecido.'}), 400

    db = load_faiss_index(context)
    if not db:
        return jsonify({'error': 'Contexto não encontrado.'}), 404

    try:
        results = search_faiss_with_embed_query(db, question, embeddings_model)
        if results:
            context_text = "\n\n".join([doc.page_content for doc in results])

            messages = [
                {"role": "system", "content": "Você é um assistente chamado BART. Responde estritamente com base no contexto fornecido e de forma curta e simpática. Caso a resposta não esteja no contexto, responda 'Não consigo responder a essa questão, experimente reformular a questão ou mudar de tema.' Não utilizes conhecimento prévio, responde apenas sobre a pergunta e contexto."},
                {"role": "system", "content": f"Contexto: {context_text}"}
            ]

            if conversation_history:
                last_interactions = conversation_history[-3:]  # Devolve as últimas 3 interações
                for pair in last_interactions:
                    messages.append({"role": "user", "content": pair['pergunta']})
                    messages.append({"role": "assistant", "content": pair['resposta']})

            messages.append({"role": "user", "content": question})

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=250,
                temperature=0.2,
                top_p=0.9
            )

            response_text = response['choices'][0]['message']['content'].strip()

            conversation_history.append({"pergunta": question, "resposta": response_text})

            # Salvar o histórico no arquivo de texto
            save_history_to_file(conversation_history)

            return jsonify({'answer': response_text, 'history': conversation_history})
        else:
            return jsonify({'answer': "Contexto relevante não encontrado.", 'history': conversation_history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

verify_and_create_db()

if __name__ == '__main__':
    app.run(debug=True)
