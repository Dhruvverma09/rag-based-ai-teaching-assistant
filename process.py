import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests
def create_embedding(text_list):
    #  https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings (request github link for more details)

    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"] 
    return embedding


def reference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False
    })
    response= r.json()
    return response
df= joblib.load("embeddings.joblib")

income_embedding= input("Enter any question that you want to ask about the video:")
question_embedding= create_embedding([income_embedding])[0]
# print(question_embedding)

# find similarity between question embedding and chunk embeddings

similarities= cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

ind=similarities.argsort()[::-1][0:5]

new_df= df.loc[ind]

prompt= f'''I am teaching data analysis using powerbi, excel and sql. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["chunk_id", "start", "end", "text"]].to_json(orient="records")}
--------------------------------------------------------
"{income_embedding}"

users asked this question related to the video chunks, you have to answer where and how much content is taught where in which video(in which video and at what timestamp) and guide the user to go to that particular video. If users asked unrelated question, tell him that you can only answer questions related to the course.
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)
    
response= reference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

