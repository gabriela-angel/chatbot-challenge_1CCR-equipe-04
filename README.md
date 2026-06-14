# 1CCR Equipe 04: Chatbot

## Integrantes:
* Gabriela Angel Silva - RM 570808
* Izabelly Menezes - RM 570673
* Marcos Paulo Sampaio - RM 573987
* Otávio Santos - RM 570225
* Tiago Muhlmann - RM 569569
* Wesley Marques - RM 573915

## O Problema Abordado

No cenário atual da mobilidade elétrica urbana, empresas e operadores comerciais enfrentam sérios gargalos na infraestrutura de recarga de veículos elétricos (EVs). O desafio técnico proposto pela **GoodWe** destaca a ausência de mecanismos digitais e inteligentes integrados nos eletropostos para resolver de forma unificada as seguintes dores:

1. **Orquestração e Balanceamento Dinâmico de Potência:** A sobrecarga da rede elétrica local quando múltiplos veículos se conectam simultaneamente, gerando quedas de energia ou lentidão crítica na recarga.
2. **Faturamento por Ciclos de Carregamento:** A complexidade em registrar, individualizar e processar cobranças financeiras e consumo energético ($kWh$) de forma transparente para faturamento B2B/B2C.
3. **Pós-Venda e Suporte Operacional Ineficiente:** Alta dependência de suporte humano para resolver falhas operacionais básicas de software ou para realizar a triagem correta de incidentes de hardware e riscos elétricos em tempo real.

---

## Proposta do Chatbot: GoodWe ChargeGrid Intelligence

Nossa solução consiste em um assistente virtual cognitivo integrado à plataforma **GoodWe ChargeGrid Intelligence**, desenvolvido especificamente para atender à persona do **Operador Comercial / Gestor de Eletropostos**. 

O chatbot atua como uma camada de inteligência operacional de primeira linha, processando consultas complexas e transformando dados técnicos brutos em respostas claras, profissionais e acionáveis.

### Escopo de Atuação e Regras de Negócio:
* **Suporte a Operações Comerciais:** Monitoramento em tempo real do status das estações (disponível, ocupado, indisponível) e sessões de recarga ativas.
* **Transparência de Faturamento:** Esclarecimento de dúvidas sobre consumo energético individualizado e faturamento por ciclos sem alucinar dados financeiros.
* **Triagem de Segurança Avançada:** Em caso de falhas físicas (ex: travas emperradas), o sistema direciona automaticamente para a manutenção especializada. Diante de riscos elétricos iminentes (ex: cabos rompidos), o chatbot emite ordens imperativas de interrupção imediata do uso, evitando acidentes.

---

## Tecnologias Selecionadas e Justificativa Técnica

Para garantir a escalabilidade e segurança exigidas pelo ecossistema GoodWe, a arquitetura do ecossistema de inteligência artificial da solução foi projetada com a seguinte pilha tecnológica:

### 1. Modelo de Linguagem (LLM): OLLama
* **Justificativa:** O projeto exige raciocínio lógico avançado para interpretação de regras de segurança elétrica e extração de intenções do usuário em linguagem natural. Esse modelos oferece janelas de contexto robustas e excelente performance em *System Prompting*, garantindo rigidez no cumprimento das diretrizes comportamentais.

### 2. Framework de Orquestração: LangChain
* **Justificativa:** O LangChain foi selecionado para conectar o modelo de linguagem às APIs de dados dos eletropostos. Ele facilitará a implementação de agentes dinâmicos (*Agents*) e memórias de conversação em tempo real (*ConversationBufferWindowMemory*), preparando a base da aplicação para a integração do padrão RAG (*Retrieval-Augmented Generation*) na Sprint 2.

### 3. Arquitetura de Validação: Python `unittest`
* **Justificativa:** A inclusão de testes unitários automatizados desde a Sprint 1 assegura que modificações futuras no prompt ou a troca de modelos não quebrem as regras de ouro do negócio (como o bloqueio imediato por falhas elétricas). Isso eleva a confiabilidade do software antes de sua exposição ao ambiente de produção.
