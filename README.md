# GoodWe ChargeGrid Intelligence — Sprint 04

**FIAP · Ciência da Computação · 1CCR · 2026.2**  
**Disciplina:** Prompt and Artificial Intelligence  
**Challenge:** EV Challenge — GoodWe

## Integrantes

| Nome | RM |
|---|---:|
| Gabriela Angel Silva | 570808 |
| Izabelly Menezes | 570673 |
| Marcos Paulo Sampaio | 573987 |
| Otávio Santos | 570225 |
| Tiago Muhlmann | 569569 |
| Wesley Marques | 573915 |

## Sobre o projeto

O GoodWe ChargeGrid Intelligence é um chatbot de suporte a temas de mobilidade elétrica, estações de recarga e gerenciamento energético relacionados à documentação disponível da GoodWe. A Sprint 04 evolui a base conversacional das Sprints 1 e 2 e a refatoração com LangChain da Sprint 3 para um pipeline **RAG (Retrieval-Augmented Generation)**, com recuperação semântica, ChromaDB persistente, citações de fonte, controles de segurança e interface web em Gradio.

O sistema deve responder perguntas factuais com base no contexto recuperado. Quando não houver evidência suficiente, deve recusar a resposta factual em vez de inventar especificações, tarifas, valores ou procedimentos.

## Funcionalidades da Sprint 04

- **Ingestão documental:** leitura de PDFs por `PyMuPDFLoader` e de arquivos `.md`/`.txt` por `TextLoader`.
- **Limpeza e chunking:** limpeza de ruídos comuns em PDFs e divisão por `RecursiveCharacterTextSplitter`.
- **Embeddings:** `nomic-embed-text` via Ollama Cloud por padrão; Ollama local pode ser configurado.
- **Banco vetorial:** ChromaDB persistente, com coleção `goodwe_knowledge_base` e IDs determinísticos para evitar duplicação ao reindexar chunks idênticos.
- **Retrieval:** busca semântica top-k, limiar mínimo de relevância, metadados de origem e página.
- **Grounding e citação:** respostas orientadas pelo contexto recuperado e citações de documento/página.
- **Prompts versionados:** `v1`, `v2` e `v3`, selecionáveis pela variável `RAG_PROMPT_VERSION`.
- **Guardrails:** validação de escopo, bloqueio de tentativas de prompt injection e filtragem de trechos suspeitos nos documentos; recusa de perguntas fora do domínio ou sem contexto suficiente.
- **Interface web:** aplicação Gradio com conversa, estado da resposta, tabela de fontes citadas e trechos recuperados.
- **Avaliação:** conjunto de 16 casos em `evals/eval_set_rag.json`; script para RAGAS e fallback manual documentado.
- **Compatibilidade:** o modo sem RAG da Sprint 3 continua disponível para comparação.

## Arquitetura

```text
Usuário
  │
  ▼
Interface Gradio (app/app.py)
  │
  ▼
MissionEngine (src/engine.py)
  ├── Guardrails: escopo, segurança e prompt injection
  ├── Memória conversacional por sessão (Sprint 3)
  └── Pipeline RAG
       ├── ChromaDB persistente
       ├── Retriever: top-k + limiar + filtro anti-injection
       ├── Contexto com metadados de fonte/página
       ├── Prompt RAG versionado (v1/v2/v3)
       └── ChatOllama + saída estruturada Pydantic
  │
  ▼
Resposta + citações visíveis + fontes/trechos recuperados
```

## Estrutura relevante

```text
.
├── app/
│   ├── app.py                 # Interface web Gradio
│   └── README.md              # Instruções específicas da interface
├── data/
│   ├── knowledge_base/        # PDFs e documentos da base
│   └── chroma_db/             # Banco vetorial persistido (gerado localmente)
├── docs/
│   ├── relatorio_evolucao.pdf # Relatório de evolução Sprint 04
│   ├── relatorio_evolucao.md  # Fonte editável do relatório
│   ├── relatorio_rag.md       # Registro de avaliação RAG
│   └── relatorio_modelos.md   # Modelos e parâmetros
├── evals/
│   ├── eval_set.json
│   ├── eval_set_rag.json      # Conjunto de avaliação da Sprint 04
│   ├── rubrica_manual.md      # Fallback do RAGAS
│   ├── run_eval.py
│   └── run_ragas.py
├── prompts/
│   ├── system_prompt.md
│   ├── system_prompt_v1.md
│   ├── rag_prompt_v1.md
│   ├── rag_prompt_v2.md
│   └── rag_prompt_v3.md
├── src/
│   ├── chain/                 # Chains e memória da Sprint 3
│   ├── guardrails/            # Escopo, segurança e prompt injection
│   ├── rag/                   # Loader, chunking, embeddings, Chroma e retriever
│   ├── schemas/               # Saída estruturada
│   └── engine.py              # Orquestração do chatbot
├── tests/
│   ├── test_sprint3.py
│   └── test_sprint4.py
├── .env.example
├── main.py                    # Interface CLI anterior
└── requirements.txt
```

## Requisitos

- Python 3.11 ou versão compatível com as dependências instaladas.
- Chave de acesso Ollama Cloud para o modelo de chat e embeddings, ou endpoint Ollama local configurado para embeddings.
- Acesso à internet quando o modelo e os embeddings forem utilizados via nuvem.

## Instalação e configuração

### 1. Criar e ativar o ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Copie `.env.example` para `.env` e configure a chave:

```env
OLLAMA_API_KEY=sua_chave_ollama
```

Variáveis opcionais:

| Variável | Padrão | Descrição |
|---|---|---|
| `RAG_MODEL` | `gpt-oss:120b` | Modelo de chat |
| `RAG_PROMPT_VERSION` | `v3` | Versão do prompt RAG (`v1`, `v2` ou `v3`) |
| `RAG_COLLECTION` | `goodwe_knowledge_base` | Coleção ChromaDB |
| `RAG_K` | `4` | Número máximo de trechos recuperados |
| `RAG_MIN_SCORE` | `0.20` | Limiar mínimo de relevância |
| `RAG_MAX_TOKENS` | `800` | Limite de tokens gerados |
| `OLLAMA_EMBED_BASE_URL` | `https://ollama.com` | Endpoint de embeddings; use `http://localhost:11434` para Ollama local |
| `GRADIO_SERVER_PORT` | `7860` | Porta da interface Gradio |

## Indexar a base de conhecimento

A indexação deve ser executada a partir da raiz do projeto:

```bash
python -m src.rag.indexer
```

O comando carrega os PDFs e arquivos `.md`/`.txt` de `data/knowledge_base/`, limpa e divide o conteúdo, gera embeddings e persiste os vetores em `data/chroma_db/`.

Parâmetros padrão: `chunk_size=800` caracteres e `chunk_overlap=120`. Para reconstruir a coleção após mudar o chunking:

```bash
python -m src.rag.indexer --chunk-size 1000 --chunk-overlap 150 --rebuild
```

Use `--rebuild` conscientemente: ele apaga a coleção selecionada antes de indexar novamente. A pasta `data/chroma_db/` é gerada localmente e não precisa ser versionada.

## Executar a interface web

Depois de configurar `.env` e indexar a base:

```bash
python app/app.py
```

Abra `http://127.0.0.1:7860` no navegador. A interface exibe a conversa, o status do processamento, as fontes citadas com página e os trechos recuperados. Para encerrar, interrompa o processo no terminal.

## Executar a versão CLI

A interface de terminal anterior permanece disponível:

```bash
python main.py
```

## Avaliação e testes

### Testes automatizados

```bash
pytest -v
```

Os testes da Sprint 4 cobrem limpeza e chunking, validação de parâmetros, persistência/idempotência do ChromaDB, recuperação com metadados, filtro de prompt injection em documentos, prompts versionados, citação de fonte, recusa por falta de evidência e preservação do modo da Sprint 3. Os testes unitários usam embeddings determinísticos e modelo falso, portanto não substituem a avaliação do modelo real.

### Avaliação RAGAS ou fallback manual

O conjunto `evals/eval_set_rag.json` possui 16 cenários: perguntas respondíveis, perguntas sem resposta na base e casos de guardrail. Exemplos de execução:

```bash
python evals/run_ragas.py --label it1_prompt_v1 --prompt-version v1
python evals/run_ragas.py --label it2_prompt_v2 --prompt-version v2
python evals/run_ragas.py --label it3_prompt_v3 --prompt-version v3
python evals/run_ragas.py --compare it1_prompt_v1 it2_prompt_v2 it3_prompt_v3
```

Os resultados são gravados em `evals/results/`. Verifique as opções disponíveis com:

```bash
python evals/run_ragas.py --help
```

Se o RAGAS não puder ser executado ou produzir mais de 30% de valores inválidos, use o fallback manual definido em `evals/rubrica_manual.md`, mantendo o mesmo conjunto de avaliação em todas as iterações. Registre os scores, a cobertura da fonte esperada, a presença de citações, recusas indevidas e latência em `docs/relatorio_rag.md`.

## Segurança e limitações

- Respostas factuais devem estar apoiadas no contexto recuperado e citar a fonte documental.
- Perguntas fora do escopo, sem contexto relevante ou com evidência insuficiente devem ser recusadas.
- O conteúdo dos documentos é tratado como não confiável; instruções encontradas dentro dele não devem alterar as regras do sistema.
- Não são inventadas especificações de produto, tarifas, valores ou procedimentos ausentes da base.
- A documentação incluída contém arquivos em português e inglês; a qualidade de recuperação de perguntas em português sobre documentos em inglês deve ser verificada na avaliação.
- PDFs sem texto extraível não são processados por OCR nesta implementação.
- A citação de página é exibida como índice do PDF acrescido de 1, conforme o código de recuperação.

---

**FIAP · EV Challenge — GoodWe · Prompt and Artificial Intelligence · Sprint 04 · 2026.2**
