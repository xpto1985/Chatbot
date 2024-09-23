# 1. Instalar Python 
Certifique-se de que o Python está instalado a partir do [site oficial](https://www.python.org/downloads/), no projeto utilizámos versão 3.10.06.

# 2. Abra o terminal e navegue até à pasta do projeto

# 3. Criar e ativar o ambiente virtual
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/macOS
source venv/bin/activate

# 4. Instalar as dependências
pip install -r requirements.txt

# 5. Download dos modelos necessários
python models/download_models.py

# 6. Criar as bases de dados FAISS
python -c "from controllers.database_controller import verify_and_create_db; verify_and_create_db()"

# 7. Executar o servidor Flask
python app.py

Aceder via browser ao link http://localhost:'porta configurada', neste caso está a default : http://localhost:5000

# 8. Para correr no terminal para debug
python chatbot_terminal.py
