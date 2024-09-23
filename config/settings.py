import os

# Flask settings
FLASK_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 5000  
}

# Caminho para a pasta estática, onde ficam os arquivos estáticos (CSS, JavaScript, imagens)
STATIC_FOLDER = os.path.join('views', 'static')
# Caminho para a pasta de templates, onde ficam os ficheiros HTML
TEMPLATES_FOLDER = os.path.join('views', 'templates')

# Caminho para o modelo LLaMA utilizado na aplicação
MODEL_PATH = "data/llms/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"

# Dicionário com os caminhos para as bases de dados FAISS, utilizados para pesquisa rápida de vetores
FAISS_PATHS = {
    '1': 'data/faiss/db_faiss_uab',
    '2': 'data/faiss/db_faiss_mpv',
    '3': 'data/faiss/db_faiss_excel'
}

# Dicionário com os caminhos para os ficheiros de contexto, que contêm informação relevante para o modelo
CONTEXT_FILES = {
    '1': 'data/context/UAB.txt',
    '2': 'data/context/MPV.txt',
    '3': 'data/context/Dados_UCS.xlsx'
}