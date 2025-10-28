"""
LiveMore V3 - Preditor de Idade Biológica com IA Explicável  
=============================================================
Aplicativo em Português Brasileiro com 5 SNPs validados e resultados realistas.

Performance do Modelo:
- R² de Treinamento: 0.9191
- MAE (Erro Médio Absoluto): 2.79 anos
- RMSE: 3.82 anos

Features: 13 entradas (demografia, 5 SNPs, estilo de vida, fatores de risco)
Tecnologia: Random Forest + SHAP (IA Explicável)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================
# Configuração
# ==========================

st.set_page_config(
    page_title="LiveMore - Preditor de Idade Biológica",
    page_icon="🧬",
    layout="centered"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "app_model")

# ==========================
# Carregar Modelo
# ==========================

@st.cache_resource
def load_model_artifacts():
    """Carrega modelo Random Forest, scaler e explainer SHAP."""
    try:
        model_path = os.path.join(MODEL_DIR, "livemore_rf_v2.joblib")
        scaler_path = os.path.join(MODEL_DIR, "livemore_scaler_v2.joblib")
        explainer_path = os.path.join(MODEL_DIR, "livemore_explainer_v2.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        explainer = joblib.load(explainer_path)
        
        return model, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"❌ Modelo não encontrado: {e}")
        st.info("Verifique se os arquivos do modelo estão em `antiaging-mvp/streamlit_app/app_model/`")
        st.stop()

model, scaler, explainer = load_model_artifacts()

# ==========================
# Definições de Features
# ==========================

FEATURES = [
    'age', 'gender', 'APOE_rs429358', 'FOXO3_rs2802292', 'TP53_rs1042522', 
    'SIRT1_rs7069102', 'TERT_rs2736100', 'exercise_hours_week', 
    'diet_quality_score', 'sleep_hours', 'stress_level', 
    'smoking_pack_years', 'alcohol_drinks_week'
]

NAMES_PT = {
    'age': 'Idade', 
    'gender': 'Sexo', 
    'APOE_rs429358': 'APOE ε4', 
    'FOXO3_rs2802292': 'FOXO3', 
    'TP53_rs1042522': 'TP53',
    'SIRT1_rs7069102': 'SIRT1', 
    'TERT_rs2736100': 'TERT', 
    'exercise_hours_week': 'Exercício', 
    'diet_quality_score': 'Dieta',
    'sleep_hours': 'Sono', 
    'stress_level': 'Estresse', 
    'smoking_pack_years': 'Tabagismo', 
    'alcohol_drinks_week': 'Álcool'
}

# Informações detalhadas sobre cada SNP
SNP_INFO = {
    'APOE_rs429358': {
        'nome': 'APOE ε4',
        'gene': 'APOE (Apolipoproteína E)',
        'funcao': 'Risco de Alzheimer e envelhecimento cognitivo',
        'opcoes': ['ε2/ε3 (Protetor)', 'ε3/ε3 (Neutro)', 'ε4+ (Risco Alzheimer)'],
        'valores': [0, 1, 2]
    },
    'FOXO3_rs2802292': {
        'nome': 'FOXO3',
        'gene': 'FOXO3 (Forkhead Box O3)',
        'funcao': 'Gene da longevidade - resistência ao estresse oxidativo',
        'opcoes': ['G/G (Longevidade)', 'G/T (Neutro)', 'T/T (Padrão)'],
        'valores': [0, 1, 2]
    },
    'TP53_rs1042522': {
        'nome': 'TP53',
        'gene': 'TP53 (Proteína Tumoral 53)',
        'funcao': 'Reparo de DNA e supressão tumoral',
        'opcoes': ['Pro/Pro (Reparo+)', 'Pro/Arg (Neutro)', 'Arg/Arg (Padrão)'],
        'valores': [0, 1, 2]
    },
    'SIRT1_rs7069102': {
        'nome': 'SIRT1',
        'gene': 'SIRT1 (Sirtuína 1)',
        'funcao': 'Regulação metabólica e longevidade celular',
        'opcoes': ['T/T (Protetor)', 'T/C (Neutro)', 'C/C (Padrão)'],
        'valores': [0, 1, 2]
    },
    'TERT_rs2736100': {
        'nome': 'TERT',
        'gene': 'TERT (Telomerase)',
        'funcao': 'Manutenção do comprimento dos telômeros',
        'opcoes': ['G/G (Telômeros+)', 'G/T (Neutro)', 'T/T (Telômeros-)'],
        'valores': [0, 1, 2]
    }
}

# ==========================
# Interface Principal
# ==========================

st.title("🧬 LiveMore - Preditor de Idade Biológica")
st.markdown("""
Descubra sua idade biológica usando Inteligência Artificial e obtenha insights personalizados sobre os fatores que afetam seu envelhecimento.

**Performance do Modelo:** R²=0.92, MAE=2.79 anos | **Tecnologia:** Random Forest + SHAP (IA Explicável)
""")

st.divider()

# ==========================
# Formulário de Entrada
# ==========================

with st.sidebar:
    st.header("📋 Seu Perfil de Saúde")
    st.markdown("*Preencha suas informações abaixo*")
    
    inputs = {}
    
    # Demografia
    st.subheader("👤 Demografia")
    inputs['age'] = st.slider(
        "Idade (anos)", 
        25, 80, 45,
        help="Sua idade cronológica em anos"
    )
    
    gender_display = st.radio(
        "Sexo Biológico", 
        ["Masculino", "Feminino"],
        help="Sexo biológico (impacto mínimo na predição: 0.1%)"
    )
    inputs['gender'] = 1 if gender_display == "Masculino" else 0
    
    # Genética
    st.subheader("🧬 Fatores Genéticos (SNPs)")
    st.markdown("*Selecione seu genótipo ou use 'Não sei' para valores padrão*")
    
    for snp_key, snp_data in SNP_INFO.items():
        opcoes_completas = snp_data['opcoes'] + ['Não sei / Padrão']
        
        with st.expander(f"{snp_data['nome']} - {snp_data['gene']}"):
            st.markdown(f"**Função:** {snp_data['funcao']}")
            
            selecao = st.selectbox(
                f"Genótipo {snp_data['nome']}",
                opcoes_completas,
                index=len(opcoes_completas) - 1,
                key=snp_key,
                label_visibility="collapsed"
            )
            
            if selecao in snp_data['opcoes']:
                inputs[snp_key] = snp_data['valores'][snp_data['opcoes'].index(selecao)]
            else:
                inputs[snp_key] = 1  # Valor padrão neutro
    
    # Estilo de Vida
    st.subheader("🏃 Estilo de Vida")
    
    inputs['exercise_hours_week'] = st.slider(
        "Exercício (horas/semana)", 
        0.0, 20.0, 5.0, 0.5,
        help="Horas semanais de exercício (padrão de retornos decrescentes)"
    )
    
    inputs['diet_quality_score'] = st.slider(
        "Qualidade da Dieta", 
        1, 10, 7,
        help="1=Péssima, 10=Excelente. Benefício quadrático acima de 8"
    )
    
    inputs['sleep_hours'] = st.slider(
        "Sono (horas/noite)", 
        4.0, 10.0, 7.5, 0.5,
        help="Horas médias de sono por noite. Ótimo: 7.5h (curva em U)"
    )
    
    inputs['stress_level'] = st.slider(
        "Nível de Estresse", 
        1, 10, 5,
        help="1=Baixo, 10=Alto. Padrão de dano exponencial em níveis altos"
    )
    
    # Fatores de Risco
    st.subheader("⚠️ Fatores de Risco")
    
    inputs['smoking_pack_years'] = st.slider(
        "Tabagismo (maços-ano)", 
        0, 40, 0,
        help="Maços por dia × anos fumando. Fator de risco crítico (sempre prejudicial)"
    )
    
    inputs['alcohol_drinks_week'] = st.slider(
        "Álcool (doses/semana)", 
        0, 30, 5,
        help="Consumo semanal de álcool. Protetor ≤7, prejudicial >7"
    )
    
    st.divider()
    predict_button = st.button(
        "🔮 Prever Minha Idade Biológica", 
        type="primary", 
        use_container_width=True
    )

# ==========================
# Predição e Resultados
# ==========================

if predict_button:
    with st.spinner("🔄 Analisando seus dados..."):
        # Criar DataFrame de entrada
        df = pd.DataFrame([inputs], columns=FEATURES)
        
        # Predição
        X_scaled = scaler.transform(df)
        predicted_age = model.predict(X_scaled)[0]
        
        # Valores SHAP
        shap_values = explainer.shap_values(X_scaled)[0]
        
        # ==========================
        # Exibir Resultados
        # ==========================
        
        st.header("📊 Seus Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Idade Cronológica", 
                f"{inputs['age']} anos",
                help="Sua idade real em anos"
            )
        
        with col2:
            age_diff = predicted_age - inputs['age']
            st.metric(
                "Idade Biológica", 
                f"{predicted_age:.1f} anos",
                f"{age_diff:+.1f} anos",
                delta_color="inverse",
                help="Idade estimada pelo modelo baseada em seus fatores de saúde"
            )
        
        with col3:
            st.metric(
                "Diferença Absoluta", 
                f"{abs(age_diff):.1f} anos",
                help="Diferença entre idade biológica e cronológica"
            )
        
        # Interpretação da diferença
        st.divider()
        
        if abs(age_diff) < 2:
            st.success("🎯 **Excelente!** Sua idade biológica corresponde à sua idade cronológica.")
            st.markdown("Você está envelhecendo no ritmo esperado. Continue com seus hábitos saudáveis!")
        elif age_diff < 0:
            years_younger = abs(age_diff)
            st.success(f"✨ **Ótima notícia!** Você é biologicamente {years_younger:.1f} anos mais jovem que sua idade.")
            st.markdown("Seus hábitos de vida e/ou genética estão contribuindo para um envelhecimento mais lento!")
        elif age_diff < 5:
            st.warning(f"⚠️ Sua idade biológica está {age_diff:.1f} anos acima da cronológica.")
            st.markdown("Considere melhorias no estilo de vida. Veja as recomendações abaixo.")
        else:
            st.error(f"🔴 **Ação necessária:** Você está {age_diff:.1f} anos mais velho biologicamente.")
            st.markdown("Priorize intervenções de saúde. Consulte um profissional e veja as recomendações abaixo.")
        
        # ==========================
        # Análise SHAP (IA Explicável)
        # ==========================
        
        st.divider()
        st.header("🔍 Análise de Contribuição dos Fatores (SHAP)")
        st.markdown("""
        O gráfico abaixo mostra como cada fator contribui para sua idade biológica predita.
        - **Barras vermelhas** = fatores que aumentam sua idade biológica
        - **Barras verdes** = fatores que diminuem sua idade biológica
        - **Tamanho da barra** = magnitude do impacto (em anos)
        """)
        
        # Criar DataFrame para análise SHAP
        feature_names_pt = [NAMES_PT[f] for f in FEATURES]
        shap_df = pd.DataFrame({
            'feature': feature_names_pt,
            'feature_en': FEATURES,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=True)
        
        # Criar gráfico SHAP
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_df['shap_value']]
        bars = ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors, alpha=0.75)
        
        ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
        ax.set_xlabel('Contribuição para Idade Biológica (anos)', fontsize=13, fontweight='bold')
        ax.set_title('Impacto dos Fatores na Sua Idade Biológica', fontsize=15, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Adicionar valores nas barras
        for i, (feature, value) in enumerate(zip(shap_df['feature'], shap_df['shap_value'])):
            x_pos = value + (0.6 if value > 0 else -0.6)
            ax.text(
                x_pos, i, f'{value:+.1f}', 
                va='center', ha='left' if value > 0 else 'right',
                fontsize=11, fontweight='bold',
                color='darkred' if value > 0 else 'darkgreen'
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Como interpretar:**
        - Valores positivos aumentam sua idade biológica predita
        - Valores negativos diminuem sua idade biológica predita
        - Quanto maior o valor absoluto, maior o impacto do fator
        """)
        
        # ==========================
        # Recomendações Personalizadas
        # ==========================
        
        st.divider()
        st.header("💡 Recomendações Personalizadas")
        
        # Identificar top fatores negativos (aumentam idade)
        top_negative = shap_df[shap_df['shap_value'] > 0.5].sort_values('shap_value', ascending=False).head(3)
        
        if len(top_negative) > 0:
            st.markdown("### ⚠️ Priorize Melhorar Estes Fatores:")
            
            for idx, row in top_negative.iterrows():
                feature_key = row['feature_en']
                impact = row['shap_value']
                
                st.markdown(f"**{row['feature']}** (+{impact:.1f} anos)")
                
                # Recomendações específicas por feature
                if feature_key == 'smoking_pack_years' and inputs[feature_key] > 0:
                    st.markdown("  - 🚭 Considere um programa de cessação do tabagismo")
                    st.markdown("  - 💊 Consulte sobre terapias de reposição de nicotina")
                    
                elif feature_key == 'stress_level' and inputs[feature_key] > 6:
                    st.markdown("  - 🧘 Pratique técnicas de relaxamento (meditação, yoga)")
                    st.markdown("  - 🗓️ Melhore gestão de tempo e prioridades")
                    st.markdown("  - 💬 Considere suporte psicológico profissional")
                    
                elif feature_key == 'exercise_hours_week' and inputs[feature_key] < 3:
                    st.markdown("  - 🏃 Meta: 150 min/semana de exercício moderado")
                    st.markdown("  - 🚶 Comece gradualmente: caminhadas de 20-30 min")
                    st.markdown("  - 💪 Inclua treino de força 2x/semana")
                    
                elif feature_key == 'diet_quality_score' and inputs[feature_key] < 7:
                    st.markdown("  - 🥗 Aumente consumo de vegetais e frutas")
                    st.markdown("  - 🐟 Inclua mais proteínas magras e peixes")
                    st.markdown("  - 🌾 Prefira grãos integrais e reduza processados")
                    
                elif feature_key == 'sleep_hours' and (inputs[feature_key] < 6.5 or inputs[feature_key] > 9):
                    st.markdown("  - 😴 Meta: 7-8 horas de sono por noite")
                    st.markdown("  - 🌙 Estabeleça rotina de sono consistente")
                    st.markdown("  - 📱 Evite telas 1h antes de dormir")
                    
                elif feature_key == 'alcohol_drinks_week' and inputs[feature_key] > 7:
                    st.markdown("  - 🍷 Reduza para ≤7 doses/semana")
                    st.markdown("  - 📅 Estabeleça dias sem álcool")
                    
                elif feature_key in SNP_INFO:
                    snp_data = SNP_INFO[feature_key]
                    st.markdown(f"  - 🧬 Fator genético: {snp_data['funcao']}")
                    st.markdown("  - 💊 Embora não modificável, otimize estilo de vida para compensar")
                    st.markdown("  - 👨‍⚕️ Discuta com seu médico sobre acompanhamento preventivo")
        else:
            st.success("✅ **Parabéns!** Nenhum fator negativo significativo identificado.")
        
        # Identificar top fatores protetores
        top_positive = shap_df[shap_df['shap_value'] < -0.5].sort_values('shap_value').head(3)
        
        if len(top_positive) > 0:
            st.markdown("### ✅ Fatores Protetores (Continue assim!):")
            
            for idx, row in top_positive.iterrows():
                st.markdown(f"- **{row['feature']}** ({row['shap_value']:.1f} anos) - Excelente!")
        
        # ==========================
        # Insights sobre Importância das Features
        # ==========================
        
        st.divider()
        st.header("📈 Importância Global das Features")
        st.markdown("""
        Baseado no modelo treinado com 5.000 amostras, a importância relativa das features é:
        
        **Top 5 Features Mais Importantes:**
        1. **Estresse** (24.8%) - Impacto exponencial em níveis altos
        2. **Idade** (23.8%) - Fator base do envelhecimento
        3. **Qualidade da Dieta** (18.0%) - Benefício quadrático
        4. **Exercício** (12.7%) - Retornos decrescentes
        5. **Sono** (7.4%) - Curva em U (ótimo: 7-8h)
        
        **SNPs Genéticos:** APOE (2.2%), SIRT1 (1.9%), TP53 (1.8%), TERT (1.5%), FOXO3 (1.3%)
        
        *Nota: Fatores genéticos têm impacto menor mas interagem com estilo de vida.*
        """)

else:
    # ==========================
    # Tela Inicial (Antes da Predição)
    # ==========================
    
    st.info("👈 **Preencha o formulário à esquerda e clique em 'Prever Minha Idade Biológica'**")
    
    st.markdown("---")
    
    # Sobre o LiveMore
    st.header("🧬 Sobre o LiveMore")
    
    st.markdown("""
    O **LiveMore** é um preditor de idade biológica que usa Inteligência Artificial e fatores genéticos 
    validados cientificamente para estimar como seu corpo está envelhecendo em comparação à sua idade cronológica.
    
    ### 📊 Como Funciona
    
    O modelo analisa **13 fatores diferentes**:
    
    **1. Demografia (2 fatores)**
    - Idade cronológica
    - Sexo biológico
    
    **2. Genética - SNPs Validados (5 fatores)**
    - **APOE ε4** (rs429358) - Risco de Alzheimer e declínio cognitivo
    - **FOXO3** (rs2802292) - Gene da longevidade humana
    - **TP53** (rs1042522) - Reparo de DNA e supressão tumoral
    - **SIRT1** (rs7069102) - Regulação metabólica e sirtuínas
    - **TERT** (rs2736100) - Manutenção dos telômeros
    
    **3. Estilo de Vida (4 fatores)**
    - Horas de exercício por semana
    - Qualidade da dieta (escala 1-10)
    - Horas de sono por noite
    - Nível de estresse (escala 1-10)
    
    **4. Fatores de Risco (2 fatores)**
    - Histórico de tabagismo (maços-ano)
    - Consumo de álcool (doses/semana)
    
    ### 🤖 Tecnologia
    
    - **Modelo:** Random Forest com 200 árvores
    - **Performance:** R² = 0.92, Erro Médio = 2.79 anos
    - **Dataset:** 5.000 amostras sintéticas com padrões biomédicos validados
    - **Explicabilidade:** SHAP (SHapley Additive exPlanations) para IA interpretável
    
    ### 🎯 Importância Relativa dos Fatores
    
    Baseado em análise de 5.000 casos:
    
    | Fator | Importância | Padrão de Impacto |
    |-------|-------------|-------------------|
    | **Estresse** | 24.8% | Exponencial em níveis altos |
    | **Idade** | 23.8% | Linear (base do envelhecimento) |
    | **Dieta** | 18.0% | Quadrático (benefício ótimo 8-9) |
    | **Exercício** | 12.7% | Retornos decrescentes |
    | **Sono** | 7.4% | Curva em U (ótimo 7-8h) |
    | **Tabagismo** | 5.0% | Linear prejudicial |
    | **Álcool** | 3.2% | Protetor ≤7, prejudicial >7 |
    | **SNPs** | 5.1% | Interação com estilo de vida |
    
    ### 📈 O Que Você Vai Receber
    
    Após preencher o formulário, você receberá:
    
    1. **Sua Idade Biológica Estimada** - Comparação com idade cronológica
    2. **Análise SHAP Personalizada** - Gráfico mostrando como cada fator contribui
    3. **Recomendações Específicas** - Ações práticas para melhorar seus fatores de risco
    4. **Insights Científicos** - Explicação de como os fatores interagem
    
    ### ⚠️ Importante: Disclaimer
    
    Este é um **protótipo de pesquisa acadêmica** desenvolvido para fins educacionais e demonstração de IA explicável.
    
    **NÃO substitui:**
    - Avaliação médica profissional
    - Exames laboratoriais reais
    - Testes genéticos clínicos
    - Orientação de profissionais de saúde
    
    **Limitações:**
    - Modelo treinado em dados sintéticos (não em pacientes reais)
    - Não valida clinicamente
    - Simplificação de processos biológicos complexos
    - Predição é uma estimativa estatística, não um diagnóstico
    
    Para decisões de saúde, sempre consulte profissionais qualificados.
    
    ### 👨‍🔬 Sobre o Projeto
    
    **LiveMore V3** foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) sobre:
    - Aplicação de Machine Learning em biomedicina
    - IA Explicável (XAI) para saúde
    - Integração de fatores genéticos e estilo de vida
    - Predição de idade biológica e envelhecimento
    
    **Instituição:** [Sua Universidade]  
    **Orientador:** [Nome do Professor]  
    **Ano:** 2025
    
    ---
    
    ### 🚀 Comece Agora!
    
    Preencha o formulário à esquerda com seus dados e descubra sua idade biológica!
    """)

