import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.oauth2 import service_account
from google.cloud import bigquery

# Configuração das Credenciais do BigQuery
credentials_dict = {
    "type": "service_account",
    "project_id": "rnn-component-life-cycle",
    "private_key_id": "60b813248c4d3301e183970c7b09500c499219fb",
    "private_key": """-----BEGIN PRIVATE KEY-----\nMIIE....<complete aqui>...""",
    "client_email": "319436388537-compute@developer.gserviceaccount.com",
    "client_id": "110841758175210010507",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/319436388537-compute@developer.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Conectar ao BigQuery
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Consulta ao BigQuery
query = """
SELECT
    data, -- Data da observação
    temperature, -- Temperatura observada
    source -- Origem dos dados (Histórico ou Projeção)
FROM
    `rnn-component-life-cycle.df_dynamox.temperature_data` -- Substitua pelo nome da sua tabela
WHERE
    spotIddesc = 'EM3401 Bomba Seccao P1' -- Filtro para o SpotId de interesse
"""
df = client.query(query).to_dataframe()

# Conversão de tipos
df['data'] = pd.to_datetime(df['data'])

# Cálculo Estatístico
mean_temp = df['temperature'].mean()
std_temp = df['temperature'].std()
upper_limit = mean_temp + 1.96 * std_temp  # Limite Superior (Z1-p)

# Identificação de Outliers
Q1 = df['temperature'].quantile(0.25)
Q3 = df['temperature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['temperature'] < Q1 - 1.5 * IQR) | (df['temperature'] > Q3 + 1.5 * IQR)]

# Exibição dos Resultados
print(f"Média da Temperatura: {mean_temp:.2f}")
print(f"Desvio Padrão da Temperatura: {std_temp:.2f}")
print(f"Limite Superior (Z1-p): {upper_limit:.2f}")
print(f"Quantidade de Outliers: {len(outliers)}")
print("\nOutliers Identificados:")
print(outliers)

# Plotagem dos Dados
plt.figure(figsize=(14, 8))

# Histograma das Temperaturas
plt.subplot(2, 1, 1)
sns.histplot(df['temperature'], bins=30, kde=True, color='blue')
plt.axvline(mean_temp, color='orange', linestyle='--', label='Média')
plt.axvline(upper_limit, color='red', linestyle='--', label='Limite Superior (Z1-p)')
plt.title("Distribuição das Temperaturas")
plt.xlabel("Temperatura")
plt.ylabel("Frequência")
plt.legend()

# Gráfico de Linhas com o Limite Superior
plt.subplot(2, 1, 2)
sns.lineplot(data=df, x='data', y='temperature', hue='source', marker='o', linestyle='-', palette='dark')
plt.axhline(upper_limit, color='red', linestyle='--', label='Limite Superior (Z1-p)')
plt.title("Temperatura ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("Temperatura")
plt.legend()

plt.tight_layout()
plt.show()
