# Relatório de Modelos e Parâmetros — Sprint 04

## Parâmetros
| Parâmetro | Valor | Justificativa |
|---|---|---|
| temperature | 0.0 | Respostas determinísticas e fundamentadas |
| top_p | 0.9 | Mantido da Sprint 3 |
| max_tokens | 800 | __ajustar se o JSON truncar__ |
| k (top-k) | 4 | __justificar com a iteração__ |

## Comparação (mesmo eval set, mesmo prompt, mesma coleção)
| Métrica | gpt-oss:120b | __2º modelo__ |
|---|---:|---:|
| Faithfulness | __ | __ |
| Answer relevancy | __ | __ |
| Fonte esperada recuperada | __ | __ |
| Recusas indevidas | __ | __ |
| Latência média (s) | __ | __ |

Comando do 2º modelo: `python evals/run_ragas.py --label it3_modelo2 --model <modelo>`.
Confirme que o modelo existe na sua conta antes de usá-lo.