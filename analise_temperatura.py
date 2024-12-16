import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.oauth2 import service_account
from google.cloud import bigquery
import numpy as np

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

# Conversão de tipos
df['data'] = pd.to_datetime(df['data'])

# Estatísticas básicas
mean_temp = df['temperature'].mean()
std_temp = df['temperature'].std()
upper_limit = mean_temp + 1.96 * std_temp  # Limite Superior (Z1-p)

# Winsorização (substituição das caudas com base nos percentis)
lower_limit = df['temperature'].quantile(0.05)  # 5º percentil
upper_limit_percentile = df['temperature'].quantile(0.95)  # 95º percentil

df['temperature_winsorized'] = np.clip(df['temperature'], lower_limit, upper_limit_percentile)

# Truncamento (remoção de valores nas caudas)
df_truncated = df[(df['temperature'] >= lower_limit) & (df['temperature'] <= upper_limit_percentile)]

# Exibir limites no terminal
print(f"Limite Inferior (5%): {lower_limit:.2f}")
print(f"Limite Superior (95%): {upper_limit_percentile:.2f}")
print(f"Número de valores após truncamento: {len(df_truncated)}")

# Visualizar a distribuição original e ajustada
plt.figure(figsize=(14, 8))

# Histograma Original
plt.subplot(2, 1, 1)
sns.histplot(df['temperature'], bins=30, kde=True, color='blue', label='Original')
plt.axvline(lower_limit, color='green', linestyle='--', label='Limite Inferior (5%)')
plt.axvline(upper_limit_percentile, color='red', linestyle='--', label='Limite Superior (95%)')
plt.title("Distribuição Original das Temperaturas")
plt.xlabel("Temperatura")
plt.ylabel("Frequência")
plt.legend()

# Histograma Winsorizado
plt.subplot(2, 1, 2)
sns.histplot(df['temperature_winsorized'], bins=30, kde=True, color='orange', label='Winsorizado')
plt.axvline(lower_limit, color='green', linestyle='--', label='Limite Inferior (5%)')
plt.axvline(upper_limit_percentile, color='red', linestyle='--', label='Limite Superior (95%)')
plt.title("Distribuição Winsorizada das Temperaturas")
plt.xlabel("Temperatura")
plt.ylabel("Frequência")
plt.legend()

plt.tight_layout()
plt.show()

