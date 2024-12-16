import streamlit as st
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account
from google.cloud import bigquery
import base64
# Configuração do Streamlit
st.set_page_config(page_title="Projeções de Modelos por SpotId", layout="wide")

# Função para adicionar imagem como plano de fundo usando URL
def set_background(image_url):
    """
    Adiciona uma imagem como plano de fundo no app Streamlit.
    :param image_url: URL da imagem de fundo.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Configuração de fundo com URL da imagem
set_background("https://raw.githubusercontent.com/CidClayQuirino/AppProjecaoMinorComponent/main/395%2001.jpg")

# Credenciais diretamente no código
credentials_dict = {
    "type": "service_account",
    "project_id": "rnn-component-life-cycle",
    "private_key_id": "60b813248c4d3301e183970c7b09500c499219fb",
    "private_key": """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDnKn9HD2nuqNfV
K0lUAf/2q8LfAbu1rJG4FVL382nwuKfBN04K8AbgTTzEzAU0A/C2NlvJXp9Q+DBe
+jcvls7Xxtht8kZNAyxYAG2/yWC10BVv4KbOaVUaLtLpK1sfT7KkczfOD1o7eRde
0chevAQ+HXwrBrKB84LmPOLKAllAZHCn5RnNOI1a+R9usRrQtTqz93jNIlSIyuff
G/oC8EId/M1IAcLf4vHJF5HyFvehu83UISZXseiniLpx99iIAhU3CBe0sEXfApSr
LWKk8fAMPxCHXGOD7501Ggob6yU0dr3bCGA2oE7SBjEmhEXg1s8R7kiCW5Wr36nw
JbuiQ+/hAgMBAAECggEAZ7j7zwitgXalEXMQve7/yNCc6a8/aG78G1sGFVdU67wP
GXUVQXcdvrspTyw0EXjLIxcU7C/W0F/sZFFlPacvPEZuijaxMJEB7qKSnAFhsXKi
i8aDUg6VLuBfplvb0RrTj1JbccwVsqXMxLDgdRFr3esg8bVZl1GKJIChSf7vySHt
zU5/WNT3FMXm/37c7fmxkMAMR/6tzoDhwXoBWfIWXfswWvoB8l1ZauZirIfoVuUa
hFprS49dnvdDD7FAU4TreLbFIDgKXdtTB32+9s/oHc12/qP6JgH3GoGmSWPFBNOc
zbLj/Wyaywc0E852eVk//6OKAyb26sYstzfIIPsLWQKBgQDvjxeXqd+2nXcPQdhu
BZ/576cdnOj8WrxTs2IEUvJkVS6aoJpGtI3Yqei1JnJgoDQk7YKTjC9pw/cX45Ni
G859LZA14qNNLxtx5LjHM4YmGB8p1cyvCcpOQCqpvvftakAfZO+G5LhbXHqUcFnT
8nkAIbCvdYQwYy3CMq1H+nMuewKBgQD3B/InV+jMEcDZyYP2+dyv88fgZxiur9pg
ivgKprddZFzmHKZHq62VtziiTXUVfi7d3wCjNs9+/ZLn4VH9KbJONz2kb8m0v6mz
XrGzK4GzpvGHEbdTlS9GhopHYtWqzfOJObMU9/0EVYIl7BEb5ykXn5PbZidHUjoV
lvO/Z+w6UwKBgH9lNUKuUA4pRzuR5Kr9ysl7rP+OhkhNaIGKj8GE6up4ckRAzEp1
kkl6cgqD26ePCqvostwMXNp+IFVP7PzrlK/1Hw/I6tcNCidTXwBwYhK4GTqCPEuJ
hVB/xzBmIirbqiYH42l1EKVlVLrD1MFl3Ps472EfuaCR3a+8i3IPulkpAoGBAOtq
7yL/bCPOZ3Ml0FV2GRK1yC3bEnKns/19cpTz/JtMIhxKAU0cFvku+xHxrzskXZWk
B/+DJItpLK8+09vn+L0BeAVVY2yVmywNelu9goWq+1I1V/iXm84iOXV+gxGb5BNK
oZfCHaYbt0RcjJGC4m5Y7ZeQ6q3VdvVMPk6Sw1VxAoGBAIvsx10SlzpTAZl+kKSF
3z40AmkOPh+L9DEi4sx2SZso2xr2qwdkvPgdJMtX854aQVexDBHQnu78mdJ43vAb
i8+SwaxJPvrn/Eq44H5xKs4WO3BOnCrtWFo670bF4cQeEE26zL0C7Q1lSvM6ozgS
H/YBhnEEyK6ziZvV7JcuwueA
-----END PRIVATE KEY-----""",
    "client_email": "319436388537-compute@developer.gserviceaccount.com",
    "client_id": "110841758175210010507",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/319436388537-compute%40developer.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Criar credenciais usando o dicionário
try:
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
except Exception as e:
    st.error(f"Erro ao configurar credenciais: {e}")
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
model_descriptions = {
    "LSTM": "**LSTM (Long Short-Term Memory)** é uma rede neural recorrente projetada para lidar com sequências temporais complexas e aprender padrões de longo prazo.",
    "GRU": "**GRU (Gated Recurrent Unit)** é uma variação simplificada da LSTM que reduz a complexidade computacional, mantendo desempenho similar para dados sequenciais.",
    "SVR": "**SVR (Support Vector Regression)** é um método de aprendizado de máquina baseado em suporte vetorial, usado para encontrar relações precisas em dados."
}
st.sidebar.info(model_descriptions.get(selected_model, ""))

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

