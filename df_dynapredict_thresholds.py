import pandas as pd
import numpy as np
from scipy.stats import norm, gamma
from google.oauth2 import service_account
from google.cloud import bigquery

# Configuração das Credenciais do BigQuery
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
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Configuração de tabelas no BigQuery
project_id = credentials.project_id
dataset_id = "df_dynamox"
main_table = f"{project_id}.{dataset_id}.df_dynapredict_main_avg"
result_table = f"{project_id}.{dataset_id}.df_dynapredict_thresholds"

# Função para definir os intervalos dinâmicos para mu e sigma
def define_intervals(data, num_points=200):
    """
    Define intervalos dinâmicos para mu e sigma com base nos dados reais de uma coluna.
    :param data: Série ou array com os valores reais de uma coluna.
    :param num_points: Número de pontos no intervalo.
    :return: Intervalos para mu_values e sigma_values.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    std_val = np.std(data)
    
    mu_start = min_val - 0.1 * (max_val - min_val)  # 10% abaixo do mínimo
    mu_end = max_val + 0.1 * (max_val - min_val)   # 10% acima do máximo
    mu_values = np.linspace(mu_start, mu_end, num_points)
    
    sigma_start = 0.1 * std_val  # Começar em 10% do desvio padrão
    sigma_end = 3 * std_val      # Terminar em 3 vezes o desvio padrão
    sigma_values = np.linspace(sigma_start, sigma_end, num_points)
    
    return mu_values, sigma_values

# Função para calcular o threshold usando o método Bayesiano
def calculate_threshold(observed_values):
    # Definir intervalos dinâmicos para mu e sigma
    mu_values, sigma_values = define_intervals(observed_values)
    
    # Priori para sigma
    alpha_prior = 2
    beta_prior = 1
    sigma_prior = gamma(a=alpha_prior, scale=1 / beta_prior)

    # Priori para mu
    mu_prior_mean = np.median(observed_values)
    mu_prior_std = 10
    mu_prior = norm(loc=mu_prior_mean, scale=mu_prior_std)

    # Construção da posterior
    log_posterior = np.zeros((len(mu_values), len(sigma_values)))

    for i, mu in enumerate(mu_values):
        for j, sigma in enumerate(sigma_values):
            prior_mu = mu_prior.pdf(mu)
            prior_sigma = sigma_prior.pdf(sigma)
            try:
                log_likelihood = np.sum(np.log(norm.pdf(observed_values, loc=mu, scale=sigma)))
            except:
                log_likelihood = -np.inf
            log_posterior[i, j] = np.log(prior_mu) + np.log(prior_sigma) + log_likelihood

    posterior = np.exp(log_posterior - np.max(log_posterior))
    posterior /= posterior.sum()

    posterior_mu = posterior.sum(axis=1)
    posterior_sigma = posterior.sum(axis=0)

    mu_posterior_mean = mu_values[np.argmax(posterior_mu)]
    sigma_posterior_mean = sigma_values[np.argmax(posterior_sigma)]

    return mu_posterior_mean + sigma_posterior_mean

# Consultar as colunas relevantes da tabela no BigQuery
query = f"""
SELECT accelerationx, accelerationy, accelerationz, temperature, velocityx, velocityy, velocityz
FROM `{main_table}`
"""
df = client.query(query).to_dataframe()

# Calcular thresholds para cada coluna
thresholds = {}
for column in df.columns:
    observed_values = df[column].dropna().values  # Remover NaN antes de calcular
    if len(observed_values) > 0:
        thresholds[column] = calculate_threshold(observed_values)

# Criar DataFrame com os resultados
thresholds_df = pd.DataFrame(list(thresholds.items()), columns=["column_name", "threshold_value"])

# Salvar os resultados no BigQuery
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
    schema=[
        bigquery.SchemaField("column_name", "STRING"),
        bigquery.SchemaField("threshold_value", "FLOAT"),
    ],
)
job = client.load_table_from_dataframe(thresholds_df, result_table, job_config=job_config)
job.result()

print(f"Thresholds salvos na tabela {result_table} com sucesso.")

