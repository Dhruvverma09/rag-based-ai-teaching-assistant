#converting the videos to mp3
import whisper 
import os
import subprocess

files= os.listdir("videos")
i=1
for file in files:
    new_file=file.split(" - ")[0]
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{i}_audio.mp3"])
    i+=1
