
#Utilities for downloading and processing of Audio and Video files
from pytube import YouTube as pyt
import os
from openai import OpenAI
import json



def download_video(url, path):
    yt = pyt(url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path=f"{path}/original_files/video", filename="video_file.mp4")
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path=f"{path}/original_files/audio", filename="audio_file.mp4")

def get_title(url):
    yt = pyt(url)
    return yt.title

def split_audio(file_path, chunk_size_mb=12, output_folder="split_chunks"):
    global split_audio_return
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    if file_size_mb <= chunk_size_mb:
        print("File size is within the limit. No need to split.")
        return
    else:
        split_audio_return = True

    clip = AudioFileClip(file_path)
    total_duration = clip.duration
    chunk_duration = (chunk_size_mb / file_size_mb) * total_duration

    # Split the audio
    start = 0
    part = 1
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        chunk = clip.subclip(start, end)
        chunk_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_part{part}.mp4")
        chunk.write_audiofile(chunk_filename, bitrate="64k", codec="aac")

        print(f"Created chunk: {chunk_filename}")

        start = end
        part += 1

    clip.close()

def process_audio_file(client, folder_path, filename, output_folder):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

        json_filename = f"{os.path.splitext(filename)[0]}_transcript.json"
        output_path = os.path.join(output_folder, json_filename)

        with open(output_path, 'w') as f:
            json.dump(transcript.segments, f, indent=4)

        print(f"Transcript for {filename} saved to {output_path}")

def create_json(split_audio_return, input_folder, output_folder="transcript_json"):
    # Transcribing with Whisper-1 & Writing to JSON File(s)

    if not os.path.exists(input_folder):
        print(f"The folder {input_folder} does not exist.")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = OpenAI()

    if not split_audio_return:
        # If audio is not split, use the path to the original file
        original_audio_path = f"{input_folder}/original_files/audio"        
        if os.path.exists(original_audio_path):
            for filename in os.listdir(original_audio_path):
                if filename.endswith(".mp4"):
                    process_audio_file(client, original_audio_path, filename, output_folder)
        else:
            print(f"The original audio folder {original_audio_path} does not exist.")
    else:
        input_folder_chunks = f'{path}/split_chunks'
        # If audio is split, iterate over the split audio files
        for filename in os.listdir(f'{path}/split_chunks'):
            if filename.endswith(".mp4"):
                process_audio_file(client, input_folder_chunks, filename, output_folder)