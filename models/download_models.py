import os
from sentence_transformers import SentenceTransformer

#Faz o download de um modelo da biblioteca Sentence Transformers e guarda-o numa pasta local.
def download_model(model_name, save_directory='data/llms'):
    model_path = os.path.join(save_directory, model_name.replace('/', '_'))
    if not os.path.exists(model_path):
        print(f"Download do modelo {model_name}...")
        model = SentenceTransformer(model_name)
        model.save(model_path)
    else:
        print(f"Modelo '{model_name}' já está na pasta '{save_directory}'.")
    return model_path

#Faz o download de um modelo GGUF a partir de um URL, se ainda não estiver guardado localmente.
def download_gguf_model_if_needed(model_url, model_filename, save_directory='data/llms'):
    import requests
    
    model_path = os.path.join(save_directory, model_filename)
    if not os.path.exists(model_path):
        print(f"Download do modelo GGUF de {model_url}...")
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, 'wb') as model_file:
            model_file.write(response.content)
        print(f"Modelo '{model_filename}' descarregado com sucesso.")
    else:
        print(f"Modelo '{model_filename}' já está presente na pasta '{save_directory}'.")
    return model_path

if __name__ == "__main__":
    sentence_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    download_model(sentence_model_name)

    model_url = "https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF/resolve/main/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    model_filename = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    download_gguf_model_if_needed(model_url, model_filename)
