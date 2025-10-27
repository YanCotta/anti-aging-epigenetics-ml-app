# PLANO DE AÇÃO ESTRATÉGICO (SPRINT TCC 20 DIAS)

### INTRODUÇÃO: O PIVÔ ESTRATÉGICO E A PRIORIZAÇÃO DE DADOS

Este documento serve como o **Plano Mestre** para um pivô estratégico e deliberado no projeto. Com um prazo final de defesa de 20 dias, abandonamos o plano original de entregar uma arquitetura de produção em larga escala (FastAPI, React, ONNX), que, conforme o `PROJECT_STATUS_OCT_2025.md`, está 0% concluída. Nosso novo objetivo é reformular a defesa de TCC como um "Pitch de Startup" para o produto "LiveMore", demonstrando agilidade e foco em validação de negócios, o que é esperado de um líder de tecnologia. O entregável central será um **MVP (Produto Mínimo Viável) em Streamlit**, focado em prototipagem rápida para validar a hipótese de negócio: o uso de IA Explicável (XAI) para demonstrar o "ROI em Saúde" para o usuário.

Contudo, uma análise rigorosa do `PROJECT_STATUS_OCT_2025.md` revelou um **bloqueador crítico** que deve ser resolvido *antes* do desenvolvimento do MVP. A validação do motor de dados "Chaos v1" (Issue #49) falhou em seu objetivo mais importante: criar complexidade não-linear. A métrica `RF vs Linear Gain: -1.82%` prova que o modelo avançado (Random Forest) que planejamos usar para XAI (SHAP) é, na verdade, *pior* do que uma simples Regressão Linear.

**Consequência Estratégica:** Construir o MVP com os dados atuais (`datasets_chaos_v1/`) resultaria em uma demonstração que *invalida* nossa própria proposta de valor. Estaríamos mostrando que nossa tecnologia "avançada" é inútil.

Portanto, a **prioridade absoluta deste sprint (Semana 1)** não é construir o aplicativo, mas sim **executar a Issue #50**. Isso envolve a refatoração do motor de geração de dados (`generator_v2_biological.py`) para injetar interações não-lineares mais fortes e variância dependente da idade, conforme especificado no relatório de status. Nosso objetivo é gerar um novo `datasets_chaos_v2/` onde o Random Forest supere o modelo linear em pelo menos 5%. Somente após a validação deste novo ativo de dados, o desenvolvimento do MVP Streamlit (Semana 2) poderá começar. Este plano detalha as tarefas exatas para o Copilot executar essa refatoração de dados primeiro e, em seguida, construir o produto de demonstração.

-----

## 1\. CONTEXTO ESTRATÉGICO (O "PORQUÊ")

### 1.1. O Pivô EstratégICO (A Nova Missão)

  * **DE (Plano Antigo):** Tentar entregar uma arquitetura de produção `FastAPI + React + ONNX + Docker` (Status atual: 0% completa, conforme `PROJECT_STATUS_OCT_2025.md`).
  * **PARA (Plano Novo):** Entregar um "Pitch de Investidor" (a defesa de TCC) com um **MVP (Produto Mínimo Viável) funcional em Streamlit**.
  * **Narrativa de Negócio (Startup "LiveMore"):** A defesa não é sobre código, é sobre *validar uma hipótese de produto*. Demonstraremos a capacidade de um líder de tecnologia em usar prototipagem rápida (Streamlit) para validar uma ideia (ROI em Saúde com XAI) antes de alocar capital de engenharia (o plano FastAPI/React).

### 1.2. ⚠️ O "BUG" CRÍTICO (A Barreira Atual)

A análise do `PROJECT_STATUS_OCT_2025.md` revela uma falha crítica nos dados atuais (`datasets_chaos_v1/`):

  * **O Problema:** O "Chaos Engine" (Issue \#49) falhou em criar complexidade não-linear.
  * **A Evidência:** `RF vs Linear Gain: -1.82%`. Um modelo Random Forest (RF) tem desempenho *pior* que uma Regressão Linear.
  * **A Ameaça:** Nossa Proposta de Valor (XAI com SHAP em um RF) é invalidada pelos nossos próprios dados. A demo do MVP falhará em justificar o uso de ML não-linear.

### 1.3. A Solução (A Prioridade Absoluta)

A primeira semana de trabalho deve ser 100% focada em consertar os dados.

  * **A Tarefa:** Implementar a **"Issue \#50"**, que é explicitamente definida no `PROJECT_STATUS_OCT_2025.md`.
  * **O Objetivo:** Criar um novo dataset (`datasets_chaos_v2/`) onde o RF supere a Regressão Linear em \>5%, validando o uso de XAI.

-----

## 2\. DEFINIÇÃO DE ENTREGÁVEIS (20 DIAS)

1.  **Entregável de Dados (Semana 1):** Um novo conjunto de dados `ml_pipeline/data_generation/datasets_chaos_v2/` que passa nos critérios de validação (RF Gain \> 5%).
2.  **Entregável de ML (Semana 2):** Artefatos de modelo (`.joblib` e `.pkl`) treinados no `datasets_chaos_v2/`.
3.  **Entregável de Produto (Semana 2):** Um app `antiaging-mvp/streamlit_app/app.py` funcional e polido que carrega os artefatos `_v2` e demonstra o "Simulador de ROI em Saúde" com SHAP.
4.  **Entregável de Negócio (Semana 3):** Um "Pitch Deck" (PowerPoint/Slides) que usa a demo do Streamlit como seu clímax.

-----

## 3\. ESCOPO NEGATIVO (O QUE IGNORAR)

O Agente Copilot deve **IGNORAR ATIVAMENTE** as seguintes pastas e tecnologias. Elas são "legado" ou "pós-TCC":

  * **NÃO MODIFICAR:**
      * `antiaging-mvp/backend/`
      * `antiaging-mvp/frontend/`
      * `antiaging-mvp/nginx/`
      * `antiaging-mvp/docker-compose.yml`
  * **NÃO IMPLEMENTAR:**
      * Qualquer código `FastAPI` ou `React`.
      * Qualquer exportação para `ONNX` ou `TorchScript`.
      * Qualquer sistema de autenticação (`JWT`, `OAuth`).
      * Qualquer conexão com banco de dados (`PostgreSQL`, `SQLite`).
      * Qualquer script `Docker`.

-----

## 4\. PLANO DE AÇÃO DETALHADO (O "COMO")

### Semana 1: A FUNDAÇÃO (Corrigindo o Ativo de Dados)

**Meta:** Implementar "Issue \#50" e gerar `datasets_chaos_v2/`.

  * **[TAREFA 1 - CRÍTICA] Modificar o Gerador de Dados**

      * **Arquivo Alvo:** `ml_pipeline/data_generation/generator_v2_biological.py`.
      * **Contexto:** O `PROJECT_STATUS_OCT_2025.md` lista explicitamente as falhas do "Chaos v1".
      * **Ação (Implementar Issue \#50):** Modificar a lógica do gerador (provavelmente na classe `ChaosConfig` ou lógica de aplicação) para:
        1.  **Aumentar `interaction_strength`** (de 1.0 para 2.5-3.0).
        2.  **Aumentar `elderly_noise_scale`** (de 6.0 para 12.0-15.0).
        3.  **Aumentar `pathway_correlation`** (de 0.4 para 0.6-0.7).
        4.  **Adicionar termos de interação não-lineares** (ex: `exp()`, `log()`, `thresholds`) em vez de apenas multiplicação simples.

  * **[TAREFA 2] Gerar e Validar `datasets_chaos_v2/`**

      * **Ação:** Rodar o `generator_v2_biological.py` modificado para gerar uma nova pasta `ml_pipeline/data_generation/datasets_chaos_v2/`.
      * **Ação de Validação (KPI):**
        1.  Copiar o notebook `notebooks/01_baseline_statistical_analysis.ipynb` para `notebooks/03_validation_chaos_v2.ipynb`.
        2.  Modificar o `03_...ipynb` para carregar os dados de `datasets_chaos_v2/`.
        3.  Rodar a análise "Non-Linearity Analysis".
        4.  **Critério de Sucesso:** `RF vs Linear Gain` deve ser **\> 5%**. Se falhar, ajustar os parâmetros da TAREFA 1 e repetir.

### Semana 2: O PRODUTO CORE (O "Demo")

**Meta:** Treinar o modelo V2 e construir o MVP Streamlit.

  * **[TAREFA 3] Refatorar Script de Treinamento**

      * **Arquivo Alvo:** Criar `ml_pipeline/train_model.py`.
      * **Arquivo Fonte (para refatorar):** `notebooks/02_random_forest_onnx_shap.ipynb`.
      * **Ações de Refatoração:**
        1.  Converter o notebook em um script Python (`.py`).
        2.  **MUDANÇA CRÍTICA:** Apontar o script para carregar dados de `ml_pipeline/data_generation/datasets_chaos_v2/train.csv`.
        3.  **MUDANÇA CRÍTICA:** **Remover** toda e qualquer lógica de exportação `ONNX`. O Streamlit usará `joblib` ou `pickle`.
        4.  Garantir que o script salve dois artefatos:
              * O modelo treinado: `antiaging-mvp/streamlit_app/app_model/livemore_rf_v2.joblib`
              * O explainer SHAP: `antiaging-mvp/streamlit_app/app_model/livemore_explainer_v2.pkl`

  * **[TAREFA 4] Refatorar e Finalizar o App Streamlit**

      * **Arquivo Alvo:** `antiaging-mvp/streamlit_app/app.py`.
      * **Ações de Refatoração:**
        1.  Garantir que o app carregue os novos artefatos (`_v2.joblib`, `_v2.pkl`).
        2.  Construir a UI na `st.sidebar` com `st.slider` e `st.selectbox` para os *principais* fatores de estilo de vida (ex: `exercise_minutes_per_week`, `smoking_pack_years`, `diet_quality_score`).
        3.  Implementar o botão "Calcular Meu ROI de Saúde".
        4.  No *main panel*, exibir:
              * `st.metric(label="Sua Idade Biológica", ...)`
              * Um gráfico SHAP (ex: `st.pyplot(shap.plots.waterfall(...))` ou `shap.plots.force(...)`) mostrando o impacto de cada *input* do usuário.

### Semana 3: O "PITCH" (A Narrativa de Negócio)

  * **[TAREFA 5] Polir Narrativa do Produto**

      * **Arquivo Alvo:** `antiaging-mvp/streamlit_app/app.py`.
      * **Ação:** Adicionar `st.title("LiveMore: Simulador de ROI em Saúde")`. Adicionar textos (`st.info`, `st.caption`) que usem linguagem de negócios (ex: "Veja como otimizar seus hábitos para o máximo retorno em longevidade.").

  * **[TAREFA 6] Criar Pitch Deck (Tarefa do Usuário)**

      * **Ação:** (Fora do Copilot) Criar a apresentação de slides focada em Problema, Solução, IP (Dados Sintéticos), Demo (Streamlit), KPIs e Roadmap Futuro (o plano FastAPI/React).

-----

## 5\. INSTRUÇÕES DE REFAÇÃO DO REPOSITÓRIO (Para Agente Copilot)

**Objetivo:** Alinhar toda a estrutura do repositório com este pivô.

1.  **`README.md` (Raiz):**

      * **Ação:** Reescrever completamente. A nova narrativa é "Startup LiveMore".
      * **Conteúdo:** "Este é o repositório do TCC 'LiveMore', um MVP de prototipagem rápida (Streamlit) para validar uma plataforma de ROI em Saúde usando IA Explicável (XAI). O objetivo é demonstrar a validação de uma hipótese de produto para uma futura arquitetura de produção."
      * **Status:** "MVP Streamlit: ATIVO. Arquitetura de Produção (FastAPI/React): ARQUIVADA (Roadmap Futuro)."
      * **Linkar para:** `antiaging-mvp/streamlit_app/README.md`.

2.  **`docs/ROADMAP.md`:**

      * **Ação:** Renomear para `docs/FUTURE_ROADMAP_POST_TCC.md`.
      * **Ação:** Adicionar um aviso no topo: `"[AVISO] Este documento descreve o roadmap de produção pós-validação do TCC. O escopo atual do TCC está focado no MVP Streamlit, descrito em MASTER_TCC_PIVOT_PLAN.md."`

3.  **`antiaging-mvp/backend/` e `antiaging-mvp/frontend/`:**

      * **Ação:** Criar um `README.md` em cada uma destas pastas.
      * **Conteúdo do README:** `"[ARQUIVADO - FORA DO ESCOPO DO TCC] Esta pasta contém o planejamento da arquitetura de produção. O trabalho atual do TCC está focado no MVP em Streamlit. Veja /docs/FUTURE_ROADMAP_POST_TCC.md para detalhes."`

4.  **`antiaging-mvp/streamlit_app/`:**

      * **Ação:** Criar (ou sobrescrever) `README.md` nesta pasta.
      * **Conteúdo:** Incluir instruções de setup limpas, sem Docker/ONNX:
        ```markdown
        # LiveMore MVP - Streamlit Demo

        Este é o MVP funcional para a defesa do TCC.

        ## Setup (Local)

        1. Crie um ambiente virtual:
           python -m venv .venv
           source .venv/bin/activate

        2. Instale as dependências:
           pip install -r requirements.txt

        3. (Pré-requisito) Treine o modelo (se os artefatos não existirem):
           # Isso usa os dados de 'datasets_chaos_v2/'
           python ../../ml_pipeline/train_model.py

        4. Rode o app:
           streamlit run app.py
        ```
      * **Ação:** Atualizar `antiaging-mvp/streamlit_app/requirements.txt`. Garantir que `onnxruntime` **NÃO** esteja lá. Incluir `streamlit`, `pandas`, `scikit-learn`, `shap`, `matplotlib`.

5.  **`notebooks/`:**

      * **Ação:** Mover os notebooks antigos para uma pasta de arquivo morto para evitar que o Copilot os use por engano.
          * Mover `notebooks/01_baseline_statistical_analysis.ipynb` -\> `legacy/notebooks-archive/01_baseline_statistical_analysis.ipynb`
          * Mover `notebooks/02_random_forest_onnx_shap.ipynb` -\> `legacy/notebooks-archive/02_random_forest_onnx_shap.ipynb`
      * **Ação:** O novo notebook `notebooks/03_validation_chaos_v2.ipynb` (da TAREFA 2) deve permanecer.

6.  **`PROJECT_STATUS_OCT_2025.md`:**

      * **Ação:** **NÃO MODIFICAR.** Este arquivo é o nosso "log histórico" e a justificativa para a Issue \#50.

7.  **`.gitignore`:**

      * **Ação:** Garantir que as seguintes linhas existam:
        ```
        # Artefatos de modelo
        *.joblib
        *.pkl

        # Datasets (muito grandes)
        ml_pipeline/data_generation/datasets_chaos_v1/
        ml_pipeline/data_generation/datasets_chaos_v2/
        ```

<!-- end list -->

```
```