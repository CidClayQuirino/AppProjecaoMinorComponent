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

# Conectar ao BigQuery
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
sigma_prior = gamma(a=alpha_prior, scale=1/beta_prior)

# Priori para mu (Normal centrada na mediana observada)
mu_prior_mean = np.median(observed_temperatures)
mu_prior_std = 10  # Variância ampla para refletir incerteza inicial
mu_prior = norm(loc=mu_prior_mean, scale=mu_prior_std)

# Função de Likelihood
def likelihood(data, mu, sigma):
    return np.prod(norm.pdf(data, loc=mu, scale=sigma))

# Construção da Posterior
mu_values = np.linspace(50, 80, 200)  # Intervalo para mu
sigma_values = np.linspace(5, 30, 200)  # Intervalo para sigma

posterior = np.zeros((len(mu_values), len(sigma_values)))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        prior_mu = mu_prior.pdf(mu)
        prior_sigma = sigma_prior.pdf(sigma)
        likelihood_value = likelihood(observed_temperatures, mu, sigma)
        posterior[i, j] = prior_mu * prior_sigma * likelihood_value

# Normalizar a posterior
posterior /= posterior.sum()

# Calcular as marginais
posterior_mu = posterior.sum(axis=1)  # Marginalizar sigma
posterior_sigma = posterior.sum(axis=0)  # Marginalizar mu

# Ajustar escala das marginais para evitar valores muito baixos
posterior_mu /= posterior_mu.max()  # Normalizar
posterior_sigma /= posterior_sigma.max()  # Normalizar

# Filtrar dados para cauda superior (com base na média e desvio padrão da posterior)
mu_posterior_mean = mu_values[np.argmax(posterior_mu)]
sigma_posterior_mean = sigma_values[np.argmax(posterior_sigma)]

# Considerar como "cauda superior" os dados acima de 1 desvio padrão da posterior
threshold = mu_posterior_mean + sigma_posterior_mean
tail_data = observed_temperatures[observed_temperatures > threshold]


for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        prior_mu = mu_prior.pdf(mu)
        prior_sigma = sigma_prior.pdf(sigma)
        likelihood_value = likelihood(observed_temperatures, mu, sigma)
        
        # Verificar valores intermediários
        if np.isnan(prior_mu) or np.isnan(prior_sigma) or np.isnan(likelihood_value):
            print(f"Valores inválidos encontrados: mu={mu}, sigma={sigma}")
            print(f"prior_mu={prior_mu}, prior_sigma={prior_sigma}, likelihood={likelihood_value}")
        
        posterior[i, j] = prior_mu * prior_sigma * likelihood_value

mu_values = np.linspace(np.min(observed_temperatures), np.max(observed_temperatures), 200)
sigma_values = np.linspace(1, np.std(observed_temperatures) * 3, 200)

log_posterior = np.zeros((len(mu_values), len(sigma_values)))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        prior_mu = mu_prior.pdf(mu)
        prior_sigma = sigma_prior.pdf(sigma)
        log_likelihood_value = np.sum(np.log(norm.pdf(observed_temperatures, loc=mu, scale=sigma)))
        
        # Evitar erros no logaritmo
        if np.isinf(log_likelihood_value):
            log_likelihood_value = -np.inf
        
        log_posterior[i, j] = np.log(prior_mu) + np.log(prior_sigma) + log_likelihood_value

# Exponenciar para retornar à escala original
posterior = np.exp(log_posterior - np.max(log_posterior))  # Normalizar para evitar overflow
posterior /= posterior.sum()  # Normalizar


# Verificar os valores calculados das marginais
print("Valores de posterior_mu (não normalizado):", posterior.sum(axis=1))
print("Valores de posterior_sigma (não normalizado):", posterior.sum(axis=0))

# Adicionar um gráfico de depuração para as marginais
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(mu_values, posterior.sum(axis=1), color="blue", label="Marginal Não Normalizada de μ")
plt.title("Marginal Não Normalizada de μ")
plt.xlabel("Média (μ)")
plt.ylabel("Densidade")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sigma_values, posterior.sum(axis=0), color="green", label="Marginal Não Normalizada de σ")
plt.title("Marginal Não Normalizada de σ")
plt.xlabel("Desvio Padrão (σ)")
plt.ylabel("Densidade")
plt.legend()

plt.tight_layout()
plt.show()

# Calcular a média das marginais para verificar os valores principais
mu_mean = np.mean(posterior_mu)
sigma_mean = np.mean(posterior_sigma)
print(f"Média da marginal de μ: {mu_mean}")
print(f"Média da marginal de σ: {sigma_mean}")

# Checar se os valores das marginais são muito pequenos
if np.all(posterior_mu < 1e-5) or np.all(posterior_sigma < 1e-5):
    print("Aviso: Os valores das marginais estão muito baixos. Considere ajustar os intervalos de mu ou sigma.")

    
