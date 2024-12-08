# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from google.cloud import bigquery
import requests
import json

# Configuração do Streamlit
st.set_page_config(page_title="Projeções de Modelos por SpotId", layout="wide")

# URL do arquivo JSON no GitHub
CREDENTIALS_URL = "https://raw.githubusercontent.com/CidClayQuirino/AppProjecaoMinorComponent/main/credentials.json"

# Carregar credenciais do GitHub
try:
    response = requests.get(CREDENTIALS_URL)
    response.raise_for_status()  # Verificar se a URL está acessível
    credentials_info = response.json()
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
except requests.exceptions.RequestException as e:
    st.error(f"Erro ao carregar as credenciais do GitHub: {e}")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Erro ao configurar as credenciais: {e}")
    st.stop()

# Configuração das tabelas e colunas relevantes
project_id = credentials.project_id
dataset_id = "df_dynamox"
main_table = f"{project_id}.{dataset_id}.df_dynapredict_main_avg"
model_tables = {
    "GRU": f"{project_id}.{dataset_id}.df_dynapredict_gru_results",
    "LSTM": f"{project_id}.{dataset_id}.df_dynapredict_lstm_results",
    "SVR": f"{project_id}.{dataset_id}.df_dynapredict_svr_results",
}
columns = ['data', 'hora', 'spotIddesc', 'accelerationx', 'accelerationy',
           'accelerationz', 'temperature', 'velocityx', 'velocityy', 'velocityz']

# Lista de SpotIds disponíveis
spot_ids = [
    "EM3401 Bomba Seccao P1",
    "EM3401 Bomba Seccao P2",
    "EM3401 Motor de Translacao LD",
    "EM3401 Motor de Translacao LE",
    "EM3401 Cilindro Boom LD",
    "EM3401 Cilindro Boom LE",
    "EM3401 Rolamento Giro Frontal",
    "EM3401 Rolamento Giro Traseiro",
    "EM3401 Motor Redutor de Giro Dianteiro",
    "EM3401 Motor Redutor de Giro Traseiro",
    "EM3401 Redutor de Giro Frontal",
    "EM3401 Redutor Giro Traseiro",
]

# Sidebar para seleção
st.sidebar.title("Configurações")
selected_item = st.sidebar.selectbox("Selecione o SpotId:", options=spot_ids)
selected_model = st.sidebar.selectbox(
    "Selecione o modelo de projeção:", options=["GRU", "LSTM", "SVR"]
)

# Breve descrição do modelo selecionado
if selected_model == "LSTM":
    st.sidebar.info("**LSTM (Long Short-Term Memory)** é uma rede neural recorrente projetada para lidar com sequências temporais complexas e aprender padrões de longo prazo.")
elif selected_model == "GRU":
    st.sidebar.info("**GRU (Gated Recurrent Unit)** é uma variação simplificada da LSTM que reduz a complexidade computacional, mantendo desempenho similar para dados sequenciais.")
elif selected_model == "SVR":
    st.sidebar.info("**SVR (Support Vector Regression)** é um método de aprendizado de máquina baseado em suporte vetorial, usado para encontrar relações precisas em dados.")

# Mensagem ao usuário sobre a seleção
st.write(f"Gerando gráficos para o SpotId: **{selected_item}** usando o modelo: **{selected_model}**")

# Consultas ao BigQuery
@st.cache_data
def carregar_dados(query):
    """Executa uma consulta no BigQuery e retorna um DataFrame."""
    query_job = client.query(query)
    return query_job.to_dataframe()

# Carregar dados históricos
query_main = f"""
SELECT {', '.join(columns)}
FROM `{main_table}`
WHERE spotIddesc = '{selected_item}'
"""
try:
    df_main = carregar_dados(query_main)
    df_main['data'] = pd.to_datetime(df_main['data'])
    df_main['source'] = 'Histórico'
except Exception as e:
    st.error(f"Erro ao carregar dados históricos: {e}")
    df_main = pd.DataFrame()

# Carregar dados do modelo selecionado
query_model = f"""
SELECT {', '.join(columns)}
FROM `{model_tables[selected_model]}`
WHERE spotIddesc = '{selected_item}'
"""
try:
    df_model = carregar_dados(query_model)
    df_model['data'] = pd.to_datetime(df_model['data'])
    df_model['source'] = f'Projeção - {selected_model}'
except Exception as e:
    st.error(f"Erro ao carregar dados do modelo {selected_model}: {e}")
    df_model = pd.DataFrame()

# Combinar dados históricos e projeções, se ambos existirem
if not df_main.empty and not df_model.empty:
    combined_df = pd.concat([df_main, df_model], ignore_index=True)

    # Reordenar as colunas para exibir 'temperature' primeiro
    ordered_columns = ['temperature'] + [col for col in columns[3:] if col != 'temperature']

    # Criar gráficos interativos
    st.subheader(f"Gráficos para o SpotId: {selected_item} usando o modelo: {selected_model}")
    for col in ordered_columns:
        fig = px.scatter(
            combined_df,
            x="data",
            y=col,
            color="source",
            title=f"{col.capitalize()} para {selected_item}",
            labels={"data": "Data", col: col.capitalize(), "source": "Origem"},
        )
        fig.update_layout(
            legend=dict(title="Origem"),
            xaxis_title="Data",
            yaxis_title=col.capitalize(),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Os dados históricos ou de projeção não foram encontrados para o SpotId ou modelo selecionado.")


