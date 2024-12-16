import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma
import seaborn as sns
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

# Consulta ao BigQuery
query = """
SELECT
    data,
    temperature
FROM
    `rnn-component-life-cycle.df_dynamox.df_dynapredict_main_avg`
WHERE
    spotIddesc = 'EM3401 Bomba Seccao P1'
"""

# Executar a consulta
df = client.query(query).to_dataframe()

# Conversão de tipos e filtragem de dados
df['data'] = pd.to_datetime(df['data'])
df = df[df['temperature'] > 0]  # Remover valores não positivos

# Dados observados
observed_temperatures = df['temperature'].values

# Configuração da Priori
# Priori para sigma (Gamma: prioriza cauda superior)
alpha_prior = 2
beta_prior = 1
sigma_prior = gamma(a=alpha_prior, scale=1 / beta_prior)

# Priori para mu (Normal centrada na mediana observada)
mu_prior_mean = np.median(observed_temperatures)
mu_prior_std = 10  # Variância ampla para refletir incerteza inicial
mu_prior = norm(loc=mu_prior_mean, scale=mu_prior_std)

# Função de Likelihood
def log_likelihood(data, mu, sigma):
    """Calcula o log-likelihood para evitar underflow."""
    return np.sum(np.log(norm.pdf(data, loc=mu, scale=sigma)))

# Construção da Posterior
mu_values = np.linspace(50, 80, 200)  # Intervalo para mu
sigma_values = np.linspace(1, 30, 200)  # Intervalo para sigma (ajustado para evitar sigma=0)

log_posterior = np.zeros((len(mu_values), len(sigma_values)))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        prior_mu = mu_prior.pdf(mu)
        prior_sigma = sigma_prior.pdf(sigma)
        try:
            log_likelihood_value = log_likelihood(observed_temperatures, mu, sigma)
        except:
            log_likelihood_value = -np.inf  # Descartar combinações inválidas
        log_posterior[i, j] = np.log(prior_mu) + np.log(prior_sigma) + log_likelihood_value

# Exponenciar e normalizar a posterior
posterior = np.exp(log_posterior - np.max(log_posterior))  # Estabilizar valores exponenciais
posterior /= posterior.sum()

# Calcular as marginais
posterior_mu = posterior.sum(axis=1)  # Marginalizar sigma
posterior_sigma = posterior.sum(axis=0)  # Marginalizar mu

# Ajustar escala das marginais
posterior_mu /= posterior_mu.max()
posterior_sigma /= posterior_sigma.max()

# Calcular médias da posterior
mu_posterior_mean = mu_values[np.argmax(posterior_mu)]
sigma_posterior_mean = sigma_values[np.argmax(posterior_sigma)]

# Definir limiar para cauda superior
threshold = mu_posterior_mean + sigma_posterior_mean
tail_data = observed_temperatures[observed_temperatures > threshold]

# Plotar o histograma dos dados filtrados da cauda superior
plt.figure(figsize=(12, 6))
sns.histplot(tail_data, bins=20, kde=True, color="blue", label="Cauda Superior")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold (μ + σ): {threshold:.2f}")
plt.title("Distribuição da Cauda Superior")
plt.xlabel("Temperatura")
plt.ylabel("Frequência")
plt.legend()
plt.show()

# Plotar as marginais
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(mu_values, posterior_mu, color="blue", label="Posterior de μ")
plt.axvline(mu_posterior_mean, color="orange", linestyle="--", label="Posterior Média")
plt.title("Posterior de μ")
plt.xlabel("Média (μ)")
plt.ylabel("Densidade (Escalonada)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sigma_values, posterior_sigma, color="green", label="Posterior de σ")
plt.axvline(sigma_posterior_mean, color="orange", linestyle="--", label="Posterior σ Média")
plt.title("Posterior de σ")
plt.xlabel("Desvio Padrão (σ)")
plt.ylabel("Densidade (Escalonada)")
plt.legend()

plt.tight_layout()
plt.show()
