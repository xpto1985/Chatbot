import json
from controllers.chatbot_controller import handle_question
from config.settings import FAISS_PATHS
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def choose_context():
    """
    Permite ao utilizador escolher o contexto que deseja utilizar para a interação com o chatbot.
    """
    print("Escolha o contexto que deseja utilizar:")
    print("1. Aspetos Administrativos da UAB")
    print("2. Modelo Pedagógico Virtual")
    print("3. Conteúdos Programáticos das UCS de 3º ano de LEI.")

    choice = input("Digite o número correspondente ao contexto: ")

    if choice == '1':
        return '1'  # Corresponde ao caminho para FAISS_PATHS['1']
    elif choice == '2':
        return '2'  # Corresponde ao caminho para FAISS_PATHS['2']
    elif choice == '3':
        return '3'  # Corresponde ao caminho para FAISS_PATHS['3']
    else:
        print("Escolha inválida.")
        return None

def run_chatbot_terminal():
    """
    Inicia o loop de interação com o chatbot através do terminal.
    """
    print("Bem-vindo ao Chatbot! Para sair, digite 'sair'.")
    
    context_key = choose_context()
    if not context_key:
        print("Contexto não selecionado corretamente. Encerrando.")
        return

    history = []
    
    while True:
        question = input("\nFaça a sua pergunta: ")

        if question.lower() == "sair":
            print("Encerrando o Chatbot. Até logo!")
            break
        elif not question:
            print("Por favor, escreva a sua pergunta corretamente.")
            continue

        # Dados para a função handle_question
        data = {
            'question': question,
            'context': context_key,
            'history': history
        }

        print("Mensagens enviadas:", data) # Apenas para debug, pode ser removido
        
        # Processa a pergunta usando handle_question do chatbot_controller
        response_data = handle_question(data)
        
        if 'error' in response_data:
            print(f"\nErro: {response_data['error']}")
        else:
            print("\nResposta:", response_data['answer'])

        # Atualiza o histórico
        history = response_data['history']

if __name__ == "__main__":
    run_chatbot_terminal()
