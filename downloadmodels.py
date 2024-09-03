import os
import requests
from sentence_transformers import SentenceTransformer

def download_sentence_transformer_model_if_needed(model_name, save_directory='models'):
    # Cria o diretório, se ele não existir
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    model_path = os.path.join(save_directory, model_name.replace('/', '_'))
    
    # Verifica se o modelo já existe
    if os.path.exists(model_path):
        print(f"O modelo '{model_name}' já está na pasta '{save_directory}'.")
        # Carregar o modelo a partir do arquivo local
        model = SentenceTransformer(model_path)
    else:
        print(f"A fazer o download de '{model_name}' para a pasta '{save_directory}'...")
        # Download do modelo da Hugging Face
        model = SentenceTransformer(model_name)
        
        # Salvar o modelo na pasta
        model.save(model_path)
        print(f"Modelo '{model_name}' descarregado e guardado em '{model_path}'.")

    return model_path


def download_gguf_model_if_needed(model_url, model_filename, save_directory='models'):
    # Cria o diretório, se ele não existir
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    model_path = os.path.join(save_directory, model_filename)
    
    # Verifica se o modelo já foi descarregado
    if os.path.exists(model_path):
        print(f"O modelo '{model_filename}' já está na pasta '{save_directory}'.")
    else:
        print(f"A fazer o download de '{model_filename}' para a pasta '{save_directory}'...")
        
        # Download do modelo a partir da URL
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Escreve o conteúdo do modelo na pasta
        with open(model_path, 'wb') as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    model_file.write(chunk)
        
        print(f"Modelo '{model_filename}' descarregado e guardado em '{model_path}'.")

    return model_path


if __name__ == "__main__":
    # Configurações do modelo sentence-transformers
    sentence_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Verifica e faz o download do modelo sentence-transformers, se necessário
    sentence_model_path = download_sentence_transformer_model_if_needed(sentence_model_name)

    # Configurações do modelo gguf
    model_url = "https://huggingface.co/models/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf/resolve/main/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    model_filename = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    
    # Verifica e faz o download do modelo gguf, se necessário
    gguf_model_path = download_gguf_model_if_needed(model_url, model_filename)
    
    print(f"Modelos prontos: {sentence_model_path} e {gguf_model_path}")
