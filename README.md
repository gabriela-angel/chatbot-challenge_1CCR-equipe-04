# 1CCR Equipe 04: Chatbot - Sprint 2

## 👤 Integrantes:
* Gabriela Angel Silva - RM 570808
* Izabelly Menezes - RM 570673
* Marcos Paulo Sampaio - RM 573987
* Otávio Santos - RM 570225
* Tiago Muhlmann - RM 569569
* Wesley Marques - RM 573915

## ⚠️ O Problema Abordado

No cenário atual da mobilidade elétrica urbana, empresas e operadores comerciais enfrentam sérios gargalos na infraestrutura de recarga de veículos elétricos (EVs). O desafio técnico proposto pela **GoodWe** destaca a ausência de mecanismos digitais e inteligentes integrados nos eletropostos para resolver de forma unificada as seguintes dores:

1. **Orquestração e Balanceamento Dinâmico de Potência:** A sobrecarga da rede elétrica local quando múltiplos veículos se conectam simultaneamente, gerando quedas de energia ou lentidão crítica na recarga.
2. **Faturamento por Ciclos de Carregamento:** A complexidade em registrar, individualizar e processar cobranças financeiras e consumo energético ($kWh$) de forma transparente para faturamento B2B/B2C.
3. **Pós-Venda e Suporte Operacional Ineficiente:** Alta dependência de suporte humano para resolver falhas operacionais básicas de software ou para realizar a triagem correta de incidentes de hardware e riscos elétricos em tempo real.

---

## 🤖 Proposta do Chatbot: GoodWe ChargeGrid Intelligence

Nossa solução consiste em um assistente virtual cognitivo integrado à plataforma **GoodWe ChargeGrid Intelligence**, desenvolvido especificamente para atender à persona do **Operador Comercial / Gestor de Eletropostos**. 

O chatbot atua como uma camada de inteligência operacional de primeira linha, processando consultas complexas e transformando dados técnicos brutos em respostas claras, profissionais e acionáveis.

### Escopo de Atuação e Regras de Negócio:
* **Suporte a Operações Comerciais:** Monitoramento em tempo real do status das estações (disponível, ocupado, indisponível) e sessões de recarga ativas.
* **Transparência de Faturamento:** Esclarecimento de dúvidas sobre consumo energético individualizado e faturamento por ciclos sem alucinar dados financeiros.
* **Triagem de Segurança Avançada:** Em caso de falhas físicas (ex: travas emperradas), o sistema direciona automaticamente para a manutenção especializada. Diante de riscos elétricos iminentes (ex: cabos rompidos), o chatbot emite ordens imperativas de interrupção imediata do uso, evitando acidentes.

---

## 🖥️ Dependências

| Pacote | Versão | Uso |
|---|---|---|
| `ollama` | 0.6.2 | Cliente para a API Ollama Cloud |
| `python-dotenv` | 1.2.2 | Carregamento de variáveis de ambiente via `.env` |
| `rich` | 15.0.0 | Formatação visual do terminal (painéis, cores) |
| `prompt-toolkit` | 3.0.52 | Input interativo com estilo no terminal |
| `pyfiglet` | 1.0.4 | Geração do banner ASCII |

---

## 🗂️ Estrutura do projeto

```
chatbot-challenge_1CCR-equipe-04/
│
├── README.md
├── main.py                  # Ponto de entrada — instancia engine e inicia CLI
├── requirements.txt         # Dependências com versões fixadas
├── .env.example             # Template de variáveis de ambiente
├── .gitignore               # Ignora .env, __pycache__, .venv
│
├── src/
│   ├── __init__.py
│   ├── ui.py                # CLI (Rich + prompt-toolkit)
│   └── engine.py            # Motor do sistema — prompt + IA
│
├── prompts/
│   └── system_prompt.md     # System prompt da IA (persona GoodWe)
│
├── tests/
│   ├── modelo_teste.md     # Descrição do teste de modelo
│   └── test_model.py     # Sistema para teste do modelo
│
│
└── assets/
    └── fluxograma.png     # Fluxograma do chatbot
```

---

## ▶️ Como executar

### Pré-requisitos

- Python 3.10 ou superior
- Conta gratuita no [Ollama Cloud](https://ollama.com) com API Key gerada

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/gabriela-angel/chatbot-challenge_1CCR-equipe-04.git
cd chatbot-challenge_1CCR-equipe-04

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as credenciais
cp .env.example .env
# Edite o arquivo .env e insira sua chave Ollama:
# OLLAMA_API_KEY=sua_chave_aqui

# 5. Execute
python main.py
```

### Comandos disponíveis na CLI

| Comando | Descrição |
|---------|-----------|
| *(qualquer pergunta)* | Gera uma resposta no contexto GoodWe ChargeGrid Intelligence |
| `teste` | Execução dos testes para validação |
| `sair` | Encerra o sistema |

---

## 🗝️ Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `OLLAMA_API_KEY` | Chave de autenticação da API Ollama Cloud |

---

## 🎯 Testes

### Executar via comando embutido

Com o chatbot rodando, digite:

```
❯ teste
```

### Executar diretamente via Python

```bash
python -m tests.test_model
```

### Casos de Teste (Sprint 1)

| # | Cenário | Objetivo |
|---|---|---|
| 1 | Gerenciamento de Potência e Demanda Energética | Verificar compreensão da orquestração dinâmica de potência sem invenção de dados técnicos |
| 2 | Cobrança e Faturamento (Registro de Ciclos) | Validar aderência ao escopo de faturamento com postura profissional |
| 3 | Status das Estações e Disponibilidade | Avaliar suporte operacional básico para persona comercial ou usuário |
| 4 | Tratamento de Falhas e Direcionamento para Manutenção Física | Verificar cumprimento da diretriz de encaminhamento para suporte técnico |
| 5 | Situação de Risco Elétrico (Segurança Extrema) | Avaliar cumprimento da diretriz crítica de segurança elétrica |

A documentação completa dos casos, com perguntas, respostas esperadas e critérios de avaliação qualitativa, está em [`tests/modelo_teste.md`](tests/modelo_teste.md).

---

## 📐 Arquitetura

O fluxo de processamento segue a seguinte sequência:

```
Usuário (CLI) → ui.py → engine.py → Ollama Cloud API (gpt-oss:120b)
                    ↑                        ↓
              Histórico de             Resposta gerada
              mensagens (10 turnos)    com contexto injetado
```

O diagrama completo está disponível em [`assets/fluxograma.png`](assets/fluxograma.png).

**Decisões técnicas relevantes:**

- O `system_prompt.md` é carregado dinamicamente a partir do sistema de arquivos, permitindo iterações rápidas sem alterar o código-fonte.
- O histórico é gerenciado como uma lista de dicionários `role/content`, truncada ao dobro de `MAX_HISTORICO` (20 mensagens), preservando as interações mais recentes.
- A temperatura do modelo é fixada em `0.3` para respostas mais determinísticas e aderentes ao escopo definido.

---

## 👍 Boas Práticas Adotadas

- **Chave de API via variável de ambiente** — nenhuma credencial exposta no código ou no repositório
- **`.gitignore` configurado** — arquivos `.env`, `__pycache__` e `.venv` excluídos do versionamento
- **Separação de responsabilidades** — `engine.py` cuida da lógica de IA; `ui.py` cuida da interface
- **System prompt em arquivo externo** — facilita iteração sem tocar no código-fonte
- **Truncagem do histórico** — evita estouro do contexto em conversas longas

---

## 🎬 Vídeo de Demonstração

🔗 https://www.youtube.com/watch?v=egjpVf5t4Kk

---

*FIAP · Ciência da Computação · EV Challenge 2026 · Prompt Engineering and Artificial Intelligence*
