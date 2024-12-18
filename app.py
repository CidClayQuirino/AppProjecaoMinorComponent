import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from google.cloud import bigquery

# Caminho relativo para o arquivo de credenciais (mesma pasta do script)
current_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(current_dir, "credentials.json")

# Carregar credenciais do arquivo
try:
    with open(credentials_path, 'r') as file:
        credentials_dict = json.load(file)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    st.success("")
except FileNotFoundError:
    st.error(f"Arquivo de credenciais não encontrado no caminho: {credentials_path}")
    st.stop()
except Exception as e:
    st.error(f"Erro ao configurar credenciais: {e}")
    st.stop()

# Configuração das tabelas e colunas relevantes
project_id = credentials.project_id
dataset_id = "df_dynamox"
main_table = f"{project_id}.{dataset_id}.df_dynapredict_main_avg"
threshold_table = f"{project_id}.{dataset_id}.df_dynapredict_thresholds"
model_tables = {
    "GRU": f"{project_id}.{dataset_id}.df_dynapredict_gru_results",
    "LSTM": f"{project_id}.{dataset_id}.df_dynapredict_lstm_results",
    "SVR": f"{project_id}.{dataset_id}.df_dynapredict_svr_results",
}
parameters = ['temperature', 'accelerationx', 'accelerationy', 'accelerationz', 'velocityx', 'velocityy', 'velocityz']

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
model_descriptions = {
    "LSTM": "**LSTM (Long Short-Term Memory)** é uma rede neural recorrente projetada para lidar com sequências temporais complexas e aprender padrões de longo prazo.",
    "GRU": "**GRU (Gated Recurrent Unit)** é uma variação simplificada da LSTM que reduz a complexidade computacional, mantendo desempenho similar para dados sequenciais.",
    "SVR": "**SVR (Support Vector Regression)** é um método de aprendizado de máquina baseado em suporte vetorial, usado para encontrar relações precisas em dados."
}
st.sidebar.info(model_descriptions.get(selected_model, ""))

# Mensagem ao usuário sobre a seleção
st.write(f"Gerando gráficos para o COMPONENTE: **{selected_item}** usando o modelo: **{selected_model}**")

# Consultas ao BigQuery
@st.cache_data
def carregar_dados(query):
    """Executa uma consulta no BigQuery e retorna um DataFrame."""
    query_job = client.query(query)
    return query_job.to_dataframe()

# Carregar dados históricos
query_main = f"""
SELECT data, {', '.join(parameters)}
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
SELECT data, {', '.join(parameters)}
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

# Carregar thresholds da tabela `df_dynapredict_thresholds`
thresholds = {}
try:
    query_thresholds = f"""
    SELECT parametro, threshold
    FROM `{threshold_table}`
    WHERE spotIddesc = '{selected_item}'
    """
    thresholds_df = carregar_dados(query_thresholds)
    thresholds = dict(zip(thresholds_df['parametro'], thresholds_df['threshold']))
except Exception as e:
    st.error(f"Erro ao carregar os thresholds: {e}")

# Combinar dados históricos e projeções, se ambos existirem
if not df_main.empty and not df_model.empty:
    combined_df = pd.concat([df_main, df_model], ignore_index=True)

    # Definir grupos de parâmetros
    parameter_groups = {
        "Acelerações": ["accelerationx", "accelerationy", "accelerationz"],
        "Temperatura": ["temperature"],
        "Velocidades": ["velocityx", "velocityy", "velocityz"]
    }

    # Criar gráficos para cada grupo de parâmetros
    st.subheader(f"Gráficos para o COMPONENTE: {selected_item} usando o modelo: {selected_model}")

    for group_name, cols in parameter_groups.items():
        # Criar DataFrame filtrado para o grupo atual
        df_group = combined_df[["data", "source"] + cols]

        # Criar gráfico
        fig = px.line(
            df_group.melt(id_vars=["data", "source"], value_vars=cols, var_name="Parâmetro", value_name="Valor"),
            x="data",
            y="Valor",
            color="Parâmetro",
            line_dash="source",
            title=f"{group_name} para {selected_item}",
            labels={"data": "Data", "Valor": "Valor", "Parâmetro": "Parâmetro"}
        )

        # Adicionar thresholds, se disponíveis
        for col in cols:
            threshold = thresholds.get(col, None)
            if threshold is not None:
                fig.add_scatter(
                    x=combined_df['data'],
                    y=[threshold] * len(combined_df),
                    mode='lines',
                    line=dict(dash='dot', color='red'),
                    name=f'Threshold {col}: {threshold:.2f}'
                )

        # Atualizar layout do gráfico
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Valor",
            hovermode="x unified"
        )

        # Renderizar o gráfico no Streamlit
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Os dados históricos ou de projeção não foram encontrados para o SpotId ou modelo selecionado.")
