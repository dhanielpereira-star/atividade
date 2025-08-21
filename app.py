import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np

# Configuração da página
st.set_page_config(page_title="Painel de Atendimento Médico", layout="wide")

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("atendimentos.csv", sep=';', encoding='latin-1')
    df.columns = df.columns.str.strip()
    return df

# Carregar os dados
df = carregar_dados()

# Título
st.title("Painel de Atendimento Médico")

# VERIFICAÇÃO DAS COLUNAS PRIMEIRO
st.sidebar.markdown("### Informações do Dataset")
st.sidebar.write(f"Total de linhas: {len(df)}")
st.sidebar.write(f"Colunas disponíveis: {df.columns.tolist()}")
st.sidebar.write("Primeiras linhas:")
st.sidebar.dataframe(df.head(3))

# Encontrar a coluna de idade automaticamente
coluna_idade = None
colunas_possiveis = ['idade', 'Idade', 'IDADE', 'age', 'Age', 'AGE']
for col in df.columns:
    if any(nome in col.lower() for nome in ['idade', 'age']):
        coluna_idade = col
        break

# Cards de métricas 
if coluna_idade:
    media_idade = df[coluna_idade].mean()
else:
    media_idade = 0
    st.error("Coluna de idade não encontrada! Verifique o arquivo CSV.")

# Encontrar colunas de atestado e síndrome respiratória
coluna_atestado = None
coluna_respiratorio = None

for col in df.columns:
    col_lower = col.lower()
    if any(nome in col_lower for nome in ['atestado', 'atest', 'certificado']):
        coluna_atestado = col
    elif any(nome in col_lower for nome in ['respirator', 'sindresp', 'sindrome']):
        coluna_respiratorio = col

# Calcular totais com verificação
if coluna_atestado and coluna_atestado in df.columns:
    total_atestados = df[df[coluna_atestado] == 1].shape[0]
else:
    total_atestados = 0
    st.warning("Coluna de atestado não encontrada ou nome diferente")

if coluna_respiratorio and coluna_respiratorio in df.columns:
    total_respiratorio = df[df[coluna_respiratorio] == 1].shape[0]
else:
    total_respiratorio = 0
    st.warning("Coluna de síndrome respiratória não encontrada ou nome diferente")

st.markdown("### Resumo dos Atendimentos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média de Idade", f"{media_idade:.1f} anos")
with col2:
    st.metric("Atestados Emitidos", total_atestados)
with col3:
    st.metric("Casos Respiratórios", total_respiratorio)

st.divider()

# Linha de gráficos
with st.container():
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        # Encontrar coluna de médico
        coluna_medico = None
        for col in df.columns:
            if any(nome in col.lower() for nome in ['medico', 'doctor', 'médico']):
                coluna_medico = col
                break
        
        if coluna_medico:
            st.markdown("Atendimento por Médico")
            fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
            sns.countplot(data=df, x=coluna_medico, ax=ax1, palette="coolwarm")
            ax1.set_xlabel("")
            ax1.set_ylabel("Qtd")
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        else:
            st.warning("Coluna 'Médico' não encontrada")

    with col_graf2:
        # Encontrar coluna de turno
        coluna_turno = None
        for col in df.columns:
            if any(nome in col.lower() for nome in ['turno', 'shift', 'periodo']):
                coluna_turno = col
                break
        
        if coluna_turno:
            st.markdown("Atendimentos por Turno")
            fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
            sns.countplot(data=df, x=coluna_turno, order=df[coluna_turno].value_counts().index, ax=ax2, palette="viridis")
            ax2.set_xlabel("")
            ax2.set_ylabel("Qtd")
            st.pyplot(fig2)
        else:
            st.warning("Coluna 'Turno' não encontrada")

# Segunda linha de gráfico
with st.container():
    col_graf3, col_graf4 = st.columns(2)

    with col_graf3:
        if coluna_respiratorio and coluna_idade:
            st.markdown("Casos Respiratórios por Idade")
            respiratorios_df = df[df[coluna_respiratorio] == 1]
            fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
            sns.histplot(respiratorios_df[coluna_idade], bins=10, kde=True, color="purple", ax=ax3)
            ax3.set_xlabel("Idade")
            ax3.set_ylabel("Casos")
            st.pyplot(fig3)
        else:
            st.warning("Dados insuficientes para gráfico de casos respiratórios")

    with col_graf4:
        # Encontrar coluna de gênero
        coluna_genero = None
        for col in df.columns:
            if any(nome in col.lower() for nome in ['genero', 'gênero', 'sexo', 'gender']):
                coluna_genero = col
                break
        
        if coluna_genero:
            st.markdown("Distribuição por Gênero")
            fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
            sns.countplot(data=df, x=coluna_genero, ax=ax4, palette="pastel")
            ax4.set_xlabel("")
            ax4.set_ylabel("Qtd")
            st.pyplot(fig4)
        else:
            st.warning("Coluna 'Gênero' não encontrada")

st.divider()

# Exportar CSV
st.markdown("### Exportar Dados")
csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
st.download_button(
    label="Baixar CSV",
    data=csv,
    file_name='atendimentos_export.csv',
    mime='text/csv',
)

st.divider()

# Análises estatísticas interativas
st.markdown("## Análises Estatísticas (Distribuições)")

# BINOMIAL - probabilidade de atestados
st.markdown("### Probabilidade de Atestados (Distribuição Binomial)")

if coluna_atestado:
    p_atestado = df[coluna_atestado].mean()

    col_a, col_b = st.columns(2)
    with col_a:
        n = st.slider("Número de pacientes simulados", min_value=5, max_value=50, value=10, step=1)
    with col_b:
        k = st.slider("Número de atestados desejados (ou mais)", min_value=1, max_value=50, value=5, step=1)

    if k > n:
        st.error("O número de atestados desejados não pode ser maior que o número de pacientes.")
    else:
        prob_5oumais = 1 - binom.cdf(k - 1, n, p_atestado)
        st.write(f"Com base em uma taxa observada de {p_atestado:.1%} de emissão de atestados,")
        st.write(f"a probabilidade de pelo menos {k} atestados em {n} pacientes é **{prob_5oumais:.2%}**.")

        # Gráfico de distribuição binomial
        probs_binom = [binom.pmf(i, n, p_atestado) for i in range(n+1)]
        fig_b, ax_b = plt.subplots(figsize=(5, 3))
        bars = ax_b.bar(range(n+1), probs_binom, color=["gray" if i < k else "orange" for i in range(n+1)])
        ax_b.set_xlabel("Número de Atestados")
        ax_b.set_ylabel("Probabilidade")
        st.pyplot(fig_b)
else:
    st.warning("Coluna de atestado não disponível para análise binomial")

st.divider()

# POISSON - probabilidade de casos respiratórios
st.markdown("### Casos Respiratórios por Turno (Distribuição de Poisson)")

if coluna_respiratorio and coluna_turno:
    casos_por_turno = df.groupby(coluna_turno)[coluna_respiratorio].sum().mean()

    k_poisson = st.slider("Número de casos respiratórios desejados (ou mais)", 
                          min_value=0, max_value=10, value=3, step=1)
    prob_3oumais = 1 - poisson.cdf(k_poisson - 1, casos_por_turno)

    st.write(f"A média de casos respiratórios por turno é **{casos_por_turno:.2f}**.")
    st.write(f"A probabilidade de pelo menos {k_poisson} casos em um turno é **{prob_3oumais:.2%}**.")

    # Gráfico da distribuição de Poisson
    max_k = 10
    probs_poisson = [poisson.pmf(i, casos_por_turno) for i in range(max_k+1)]
    fig_p, ax_p = plt.subplots(figsize=(5, 3))
    bars_p = ax_p.bar(range(max_k+1), probs_poisson, color=["gray" if i < k_poisson else "orange" for i in range(max_k+1)])
    ax_p.set_xlabel("Número de Casos")
    ax_p.set_ylabel("Probabilidade")
    ax_p.set_title("Distribuição de Poisson")
    st.pyplot(fig_p)
else:
    st.warning("Dados insuficientes para análise de Poisson")