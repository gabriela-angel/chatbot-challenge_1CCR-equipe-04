# Interface web (Gradio)

1. Ative o ambiente virtual e instale `requirements.txt`.
2. Configure `.env` (veja `.env.example`) e indexe a base: `python -m src.rag.indexer`.
3. Na raiz do projeto: `python app/app.py` e abra http://127.0.0.1:7860.

Variáveis opcionais: `RAG_PROMPT_VERSION`, `RAG_COLLECTION`, `RAG_K`,
`RAG_MIN_SCORE`, `RAG_MODEL`, `RAG_MAX_TOKENS`, `GRADIO_SERVER_PORT`.
A citação aparece no texto da resposta, na tabela "Fontes citadas" e nos trechos recuperados.
Erros internos são gravados em `chatbot.log`.