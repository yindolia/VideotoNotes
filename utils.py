# Other functions to process Json and input and output of LLMs
import os
import re
import json


def extract_part_number(filename):
    match = re.search(r'part(\d+)', filename)
    return int(match.group(1)) if match else 0

def process_file(filepath, max_id, last_end_time):
    with open(filepath, 'r') as file:
        data = json.load(file)
        time_adjustment = last_end_time - float(data[0]['start']) if last_end_time else 0
        new_data = []
        for entry in data:
            new_entry = {
                'id': max_id + 1,
                'start': float(entry['start']) + time_adjustment,
                'end': float(entry['end']) + time_adjustment,
                'text': entry['text']
            }
            new_data.append(new_entry)
            max_id += 1
        return new_data, max_id, new_data[-1]['end'] if new_data else last_end_time

def process_transcripts(directory_path, split_audio_return):
    files = os.listdir(directory_path)
    combined_data = []
    max_id = -1
    last_end_time = 0.0

    if split_audio_return:
        sorted_files = sorted(
            [file for file in files if file.startswith('audio') and file.endswith('.json')],
            key=extract_part_number
        )
    else:
        sorted_files = [file for file in files if file.endswith('.json') and not 'part' in file]

    for filename in sorted_files:
        full_path = os.path.join(directory_path, filename)
        processed_data, max_id, last_end_time = process_file(full_path, max_id, last_end_time)
        combined_data.extend(processed_data)

    # Output the combined data to a new JSON file
    output_path = os.path.join(directory_path, 'combined_data.json')
    with open(output_path, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

def full_transcript(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    transcript = ""
    for entry in data:
        transcript += entry['text']
    return transcript


def merge_short_documents(documents, min_length=2000):
    i = 0
    while i < len(documents) - 1:
        current_doc = documents[i]
        if len(current_doc.page_content) < min_length:
            documents[i + 1].page_content = current_doc.page_content + documents[i + 1].page_content
            del documents[i]
        else:
            i += 1
    return documents

def generate_markdown(merged_docs, path, guide_chain):
    markdown_outputs = []
    for doc in merged_docs:
        output = guide_chain.invoke(doc.page_content)
        markdown_outputs.append(output)
    combined_output = '\n\n'.join(markdown_outputs)
    with open(f'{path}/transcript_json/llm_outline.txt', 'w') as file:
        file.write(combined_output)


def grab_frame(video, second):
    frames_dir = 'frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = round(int(second * fps))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_number >= total_frames:
        print(f"Error: Frame number {frame_number} exceeds total frames in video.")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return None

    frame_path = os.path.join(frames_dir, f'frame_{second}.jpg')
    cv2.imwrite(frame_path, frame)
    cap.release()

    return frame_path

def retrieve_time(segment):
    docs = retriever.get_relevant_documents(segment)
    docs_dict = json.loads(docs[0].page_content)
    start_time = docs_dict["start"]
    end_time = docs_dict["end"]
    time = (start_time + end_time) / 2
    final_time = round(time)
    return final_time

def create_hyperlink(segment, url):
    time = retrieve_time(segment)
    time_link = f"{url}&t={time}s"
    return time_link

def format_seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_placeholder(placeholder):
    if placeholder.startswith("<PICTURE:"):
        description = placeholder[9:-1]
        time = retrieve_time(description)
        image_path = grab_frame(video_path, time)
        # Embed the image using markdown with a specified width
        return f'<img src="{image_path}" alt="{description}" width="450"/>'
    elif placeholder.startswith("<HYPERLINK:"):
        text = placeholder[11:-1]
        time = retrieve_time(text)
        formatted_time = format_seconds_to_hms(time)
        hyperlink = create_hyperlink(text, url)
        return f'[Jump to this part of the video: {formatted_time}]({hyperlink})'
    else:
        return placeholder

def replace_placeholders(content):
    placeholders = re.findall(r"<[^>]+>", content)
    for placeholder in placeholders:
        replacement = process_placeholder(placeholder)
        content = content.replace(placeholder, replacement, 1)
    return content

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def convert_txt(path, title, db):
    txt_file_path = f'{path}/transcript_json/llm_outline.txt'
    output_file_path = f'{path}/companion_guide.txt'
    global video_path
    video_path = f'{path}/original_files/video/video_file.mp4'
    global retriever
    retriever = db.as_retriever(search_kwargs={"k": 1})
    
    content = read_file(txt_file_path)
    updated_content = replace_placeholders(content)
    
    with open(output_file_path, 'w') as file:
        file.write(updated_content)
    
    print(f"Updated markdown content has been written to {output_file_path}")