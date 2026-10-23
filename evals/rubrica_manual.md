# Rubrica manual (fallback do RAGAS) — Sprint 04

Usada apenas quando o RAGAS não puder ser executado ou retornar mais de 30%
de valores inválidos. Aplicada ao **mesmo eval set** (`evals/eval_set_rag.json`)
em **todas** as iterações. O avaliador é o próprio LLM configurado em
`--judge-model`, com temperatura 0, seguindo exatamente os critérios abaixo.

## Faithfulness (0–1)

Equivalente à definição do RAGAS: fração das afirmações factuais da resposta
que são sustentadas pelo contexto recuperado.

1. Extrair as afirmações factuais da resposta (ignorar saudações e marcadores de citação).
2. Contar quantas são diretamente sustentadas pelo contexto recuperado.
3. `faithfulness = afirmações_suportadas / afirmações_totais`.

Afirmações corretas no mundo real, mas ausentes do contexto, **não** contam como
suportadas. Resposta sem afirmações factuais não recebe nota (excluída da média).

## Answer relevancy (0–1)

Equivalente funcional à métrica do RAGAS (que mede o quanto a resposta se
mantém no tema da pergunta). Escala discreta:

| Nota | Critério |
|---|---|
| 1.00 | Responde completamente à pergunta, sem desvios |
| 0.75 | Responde ao essencial, com omissão pequena ou informação irrelevante leve |
| 0.50 | Responde parcialmente |
| 0.25 | Toca no tema, mas quase não responde |
| 0.00 | Não responde ou responde outro assunto |

## Escopo da média

Faithfulness e answer relevancy são calculadas sobre perguntas `answerable`
efetivamente respondidas. Recusas indevidas aparecem em `false_refusal_rate`;
recusas corretas (`unanswerable` e `guardrail`) aparecem em `correct_refusal_rate`.