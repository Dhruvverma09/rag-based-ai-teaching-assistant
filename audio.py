import whisper
import os
import json
model= whisper.load_model("large-v2")

audios= os.listdir("audios")
for audio in audios:
    result= model.transcribe(audio=f"audios/{audio}", language="hi", task= "translate", word_timestamps=False)
    
    chunks=[]
    for segment in result["segments"]:
        chunks.append({"start":segment["start"], "end":segment["end"], "text":segment["text"]})
        
        chunks_meta= {"chunks": chunks, "text": result["text"]}
        
        with open("output.json", "w") as f:
            json.dump(chunks_meta,f)