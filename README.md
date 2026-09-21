# 1CCR Equipe 04: Chatbot - Sprint 3

## 👤 Integrantes:

| Nome | RM |
|----------------------|--------|
| Gabriela Angel | 570808 |
| Izabelly Menezes | 570673 |
| Marcos Sampaio | 573987 |
| Otávio Santos | 570225 |
| Tiago Muhlmann | 569569 |
| Wesley Marques | 573915 |

---

## 🤖 GoodWe ChargeGrid Intelligence

Chatbot com IA voltado ao domínio de mobilidade elétrica, estações de recarga e gerenciamento energético da plataforma GoodWe ChargeGrid Intelligence.

Na **Sprint 3**, o núcleo conversacional foi refatorado para **LangChain LCEL**, incorporando:

- `ChatPromptTemplate | ChatOllama | PydanticOutputParser`;
- memória por sessão com `RunnableWithMessageHistory`;
- `ConversationTokenBufferMemory` com limite de 2.500 tokens;
- structured output com Pydantic v2 e `field_validator`;
- system prompt versionado com XML tagging;
- medição de tokens com `tiktoken`;
- guardrails para prompt injection/jailbreak, riscos elétricos, manutenção física e escopo GoodWe;
- avaliação automatizada em `evals/`.

---

## 🧩 Arquitetura da Sprint 03

```text
Usuário
   ↓
Guardrails
   ├── Prompt Injection / Jailbreak
   ├── Risco elétrico / manutenção
   └── Escopo GoodWe
   ↓
RunnableWithMessageHistory
   ↓
ChatPromptTemplate
   ↓
ChatOllama (gpt-oss:120b)
   ↓
PydanticOutputParser
   ↓
ConsultaRecarga
   ↓
Resposta estruturada + histórico da sessão
```

---

## 📁 Estrutura do projeto

```text
chatbot-challenge_1CCR-equipe-04/
│
├── README.md
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── prompts/
│   ├── system_prompt.md
│   └── system_prompt_v1.md
│
├── src/
│   ├── chain/
│   │   ├── builder.py
│   │   ├── memoria.py
│   │   └── multi_model.py
│   ├── guardrails/
│   │   ├── moderation.py
│   │   └── scope_validator.py
│   ├── schemas/
│   │   └── consulta_recarga.py
│   ├── utils/
│   │   └── tokens.py
│   ├── engine.py
│   └── ui.py
│
├── evals/
│   ├── eval_set.json
│   ├── run_eval.py
│   └── sprint3_results.json
│
├── tests/
│   ├── conftest.py
│   ├── test_sprint3.py
│   ├── test_model.py
│   └── modelo_teste.md
│
├── docs/
│   ├── relatorio_evolucao.pdf
│   └── relatorio_modelos.md
│
│
└── assets/
    └── fluxograma.png
```

---

## ⚙️ Dependências principais

| Pacote | Uso |
|---|---|
| `langchain` | Orquestração LangChain |
| `langchain-core` | LCEL, prompts e runnables |
| `langchain-ollama` | Integração com ChatOllama |
| `pydantic` | Structured output e validação |
| `tiktoken` | Medição de tokens |
| `pytest` | Testes automatizados |
| `python-dotenv` | Variáveis de ambiente |

---

## ▶️ Como executar

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar ambiente

Copie `.env.example` para `.env` e configure:

```text
OLLAMA_API_KEY=sua_chave
```

A chave não deve ser versionada.

### 4. Executar o chatbot

```bash
python main.py
```

---

## 🧪 Testes

Executar a suíte automatizada:

```bash
pytest -v
```

A suíte da Sprint 03 valida:

- Pydantic e `field_validator`;
- rejeição de valores inválidos;
- bloqueio de prompt injection;
- memória em três turnos.

---

## 📊 Avaliação da Sprint 03

A avaliação automatizada foi executada com o `evals/eval_set.json`.

Resultado registrado:

| Métrica | Resultado |
|---|---:|
| Casos avaliados | 7 |
| Casos aprovados | 6 |
| Acurácia por keyword | 85,7% |
| Structured Output Accuracy | 100% |
| Latência média | 1,497 s |
| Casos de memória | 1 |
| Memória respeitou limite | Sim |

Os resultados detalhados são armazenados em:

```text
evals/sprint3_results.json
```

---

## 🛡️ Guardrails

O sistema possui validações antes da chamada ao modelo para:

- prompt injection e jailbreak;
- riscos elétricos;
- manutenção física;
- assuntos fora do escopo GoodWe.

Em situações de risco elétrico ou manutenção física, o sistema orienta a interrupção/evitação da intervenção e o contato com suporte técnico ou profissional habilitado.

---

## 🧠 Structured Output

O schema `ConsultaRecarga` utiliza Pydantic v2 e valida:

- resposta;
- categoria;
- estado do carregador;
- potência;
- consumo;
- faturamento;
- necessidade de suporte.

Valores negativos de potência, consumo ou faturamento são rejeitados por `field_validator`.

---

## 🔢 Configuração principal do modelo

O núcleo da aplicação utiliza:

| Parâmetro | Valor |
|---|---:|
| Modelo | `gpt-oss:120b` |
| Temperature | `0.3` |
| Top-p | `0.9` |
| Max tokens | `800` |
| Limite de memória | `2500 tokens` |

A comparação de modelos e parâmetros está documentada em `docs/relatorio_modelos.md`.

---

## 📄 Documentação da Sprint 3

O relatório de evolução da Sprint 3 está em `docs/relatorio_evolucao.pdf` e apresenta:

1. resumo da evolução;
2. decisões da refatoração para LCEL;
3. comparativo antes/depois;
4. problemas encontrados e soluções;
5. equipe e divisão de trabalho.

---

## 🔐 Boas práticas

- API key somente por variável de ambiente;
- `.env` ignorado pelo Git;
- separação entre UI, engine, chain, schemas e guardrails;
- prompt externo e versionado;
- memória limitada por tokens;
- saída estruturada validada;
- testes automatizados.

---

**FIAP · Ciência da Computação · EV Challenge 2026 · Prompt and Artificial Intelligence · Sprint 3**
