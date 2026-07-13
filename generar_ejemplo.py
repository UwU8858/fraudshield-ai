"""
Genera un ejemplo de transacción cruda (JSON) para probar la API.
"""
import pandas as pd
import json

train_transaction = pd.read_csv("data/raw/train_transaction.csv")
train_identity = pd.read_csv("data/raw/train_identity.csv")

df = train_transaction.merge(train_identity, on='TransactionID', how='left')

# Tomamos una fila real, sin la etiqueta (isFraud no debe mandarse a la API)
fila_ejemplo = df.iloc[[100]].drop(columns=['isFraud']).to_dict(orient='records')[0]

# Reemplazamos NaN por None (JSON no entiende NaN)
fila_ejemplo = {k: (None if pd.isna(v) else v) for k, v in fila_ejemplo.items()}

payload = {"datos": fila_ejemplo}

with open('ejemplo_transaccion.json', 'w') as f:
    json.dump(payload, f, indent=2)

print("Ejemplo guardado en ejemplo_transaccion.json")
print(f"TransactionID de ejemplo: {fila_ejemplo['TransactionID']}")