# Relatório RAG — Sprint 04

## Base de conhecimento
| Documento | Páginas | Tipo |
|---|---:|---|
| CFT-Cartilha-Eletrobilidade-mar2026-4.pdf | 24 | Normativo/orientativo (PT) |
| GW_ESA 261kWh_Maintenance Manual-EN.pdf | 117 | Manual de produto (EN) |
| GW_ESS Troubleshooting Guide-EMEA_EN.pdf | 37 | Guia técnico (EN) |
| GW_Third-Party-EMS-Protocol_MODBUS-EN.pdf | 27 | Protocolo (EN) |
| __adicionar: manuais ChargeGrid/EV ChargeOps, regimentos, FAQs, tarifas__ | | |

## Pipeline e decisões de chunking
- Loader: PyMuPDFLoader (+ TextLoader para .md/.txt) → limpeza (`clean_text`) → RecursiveCharacterTextSplitter.
- Embeddings: nomic-embed-text; ChromaDB persistente, distância de cosseno, IDs determinísticos.
- Parâmetros testados: `__size/overlap__` → `__n chunks__`. Trade-offs: __preencher__.
- Retrieval: top-k=`__`, limiar `__` (calibrado em: __). Anti-injection: trechos suspeitos são descartados.
- Limitações conhecidas: perguntas em PT sobre documentos em EN; páginas citadas = índice do PDF + 1.

## Versões do prompt RAG
| Versão | O que mudou | Faithfulness | Answer relevancy | Citação | Ganho vs anterior |
|---|---|---:|---:|---:|---|
| v1 | Fundamentação + citação + segurança básica | __ | __ | __ | baseline |
| v2 | XML tags, frase padrão de recusa, regras p/ campos numéricos, política de manutenção | __ | __ | __ | __ |
| v3 | Procedimento passo a passo, resposta parcial, autoverificação de citações | __ | __ | __ | __ |

## Iterações (evals/results/)
| Iteração | Mudança | Prompt | Chunk | k | Faith. | Ans. rel. | Fonte esperada recuperada |
|---|---|---|---|---:|---:|---:|---:|
| baseline_sprint3 | Sem RAG | system_prompt.md | — | — | n/d | __ | n/d |
| it1 | | v1 | 800/120 | 4 | __ | __ | __ |
| it2 | | v2 | 800/120 | 4 | __ | __ | __ |
| it3 | | v3 | 1000/150 | 5 | __ | __ | __ |

Backend das métricas: RAGAS ou rubrica manual (`evals/rubrica_manual.md`), o mesmo em todas as iterações.

## Segurança
Grounding obrigatório, citação, recusa fora do contexto (escopo + limiar + frase de evidência insuficiente),
injection via documentos (filtro no retriever + prompt) e recusa de manutenção física.