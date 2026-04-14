import requests
import json
import pandas as pd
import numpy as np
import joblib
def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

my_dicts = []
chunk_id = 0

with open("output.json", "r") as f:
    content = json.load(f)
    print("Creating Embeddings for chunks...")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
        

df = pd.DataFrame.from_records(my_dicts)
print(df)
# df.to_csv("chunk.csv") for checking the data.
joblib.dump(df, "embeddings.joblib")
