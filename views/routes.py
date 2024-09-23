from flask import Flask, request, jsonify, send_from_directory
from controllers.chatbot_controller import handle_question
from config.settings import STATIC_FOLDER, TEMPLATES_FOLDER

# Inicializa as rotas para a aplicação Flask.
def initialize_routes(app):

    # Rota principal que serve a página inicial (index.html)
    @app.route('/')
    def serve_index():
        return send_from_directory(TEMPLATES_FOLDER, 'index.html')

    # Rota para servir arquivos CSS estáticos
    @app.route('/static/css/<path:filename>')
    def serve_css(filename):
        return send_from_directory(f'{STATIC_FOLDER}/css', filename)

    # Rota para servir arquivos JavaScript estáticos
    @app.route('/static/js/<path:filename>')
    def serve_js(filename):
        return send_from_directory(f'{STATIC_FOLDER}/js', filename)

    # Rota para lidar com as perguntas enviadas pelo chatbot
    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.json
        response = handle_question(data)
        return jsonify(response)
