# Relatório de Modelos — Sprint 03

## Modelos

| Modelo | Temperature | Top-p | Max tokens |
|---|------------:|------:|-----------:|
| `gpt-oss:120b` |         0.3 |   0.9 |        800 |
| `qwen3:8b` |         0.3 |   0.9 |        800 |

## Modelo principal

O modelo principal da aplicação é `gpt-oss:120b`, conforme a configuração de `src/chain/builder.py` e `src/engine.py`.

## Resultado da avaliação Sprint 03

- Casos avaliados: 7
- Aprovados: 6
- Acurácia por keyword: 85,7%
- Structured Output Accuracy: 100%
- Latência média: 1,497 s
- Memória respeitou o limite: sim
