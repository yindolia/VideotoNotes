from moviepy.editor import *
import os
import time
import json
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import cv2

import avutils as av
import utils as ut


def start_timer():
    global start_time
    start_time = time.time()

def show_current_runtime():
        return round(time.time() - start_time, 2)

start_timer()

guide_prompt_template = """

Below is a script from a video that I am making into a companion guide blog post first. \
You are a helpful assistant made to assist in the creation I'm doing. \
This is a continuation of a guide so include chapters, key summaries, and incorporate visual aids and direct links to relevant parts of the video, \
however do not include any conclusion or overarching title. \
For visual aids, specific frames from the video will be identified where images can be inserted to enhance understanding. \
For direct links, portions of the text should be hyperlinked to their corresponding times in the video. \
To indicate that a sentence should be hyperlinked, insert the raw text of the transcript next to the word with the indicator <HYPERLINK: "corresponding transcript text">. \
To indicate a picture regarding the text, insert the indicator <PICTURE: "corresponding transcript text">. \
It is crucial to use the raw text from the transcript that will be used, as the additional tools that will be inserting the hyperlinks and pictures need this to know where in the video to look.

In this blog post, in addition to the paragraphs: \

Create titles or headings that encapsulate main points and ideas \

Format your response in markdown, ensuring distinction and clean styling between titles and paragraphs. \
Be sure to include the image placeholders, and hyperlinks with enough distinguishable text WITHOUT ANY QUOTATIONS, as the placeholders will be fed into a semantic search algorithm. \
This structured approach will be applied to the entire transcript. \
The example below only shows one style, but use multiple styles including different headings, bullet points, and other markdown elements when needed. \

Here are shortened example of the input and shortened expected output:

example input:

Hi everyone. So in this video I'd like us to cover the process of tokenization in large language models. Now you see here that I have a sad face and that's because tokenization is my least favorite part of working with large language models but unfortunately it is necessary to understand in some detail because it is fairly hairy, gnarly and there's a lot of hidden foot gums to be aware of and a lot of oddness with large language models typically traces back to tokenization. So what is tokenization? Now in my previous video Let's Build GPT from Scratch we actually already did tokenization but we did a very naive simple version of tokenization. So when you go to the Google Colab for that video you see here that we loaded our training set and our training set was this Shakespeare dataset. Now in the beginning the Shakespeare dataset is just a large string in Python it's just text and so the question is how do we plug text into large language models and in this case here we created a vocabulary of 65 possible characters that we saw occur in this string. These were the possible characters and we saw that there are 65 of them and then we created a lookup table for converting from every possible character a little string piece into a token an integer. So here for example we tokenized the string hi there and we received this sequence of tokens and here we took the first 1000 characters of our dataset and we encoded it into tokens and because this is character level we received 1000 tokens in a sequence so token 18, 47, etc. Now later we saw that the way we plug these tokens into the language model is by using an embedding table and so basically if we have 65 possible tokens then this embedding table is going to have 65 rows and roughly speaking we're taking the integer associated with every single token we're using that as a lookup into this table and we're plucking out the corresponding row and this row is trainable parameters that we're going to train using backpropagation and this is the vector that then feeds into the transformer and that's how the transformer sort of perceives every single token. So here we had a very naive tokenization process that was a character level tokenizer

example output:

Introduction to Tokenization
----------------------------

Welcome to our comprehensive guide on tokenization in large language models (LLMs). Tokenization is a critical yet complex aspect of working with LLMs, essential for understanding how these models process text data. Despite its challenges, tokenization is foundational, as it converts strings of text into sequences of tokens, small units of text that LLMs can manage more effectively.

<PICTURE: Now you see here that I have a sad face and that's because tokenization is my least favorite part of working with large language models but unfortunately it is necessary to understand in some detail because it is fairly hairy, gnarly and there's a lot of hidden foot gums>

Understanding the Basics of Tokenization
----------------------------------------

Tokenization involves creating a vocabulary from all unique characters or words in a dataset and converting each into a corresponding integer token. This process was briefly introduced in our "Let's Build GPT from Scratch" video, where we tokenized a Shakespeare dataset at a character level, creating a vocabulary of 65 possible characters.

<HYPERLINK: So what is tokenization? Now in my previous video Let's Build GPT from Scratch we actually already did tokenization but we did a very naive simple version of tokenization. So when you go to the Google Colab for that video you see here that we loaded>

The Role of Embedding Tables in Tokenization
--------------------------------------------

After tokenization, the next step involves using an embedding table, where each token's integer is used as a lookup to extract a row of trainable parameters. These parameters, once trained, feed into the transformer model, allowing it to perceive each token effectively.

<PICTURE: using backpropagation and this is the vector that then feeds into the transformer and that's how the transformer sort of perceives every single token. So here we had a very naive tokenization process that was a character level tokenizer>

end examples.

Here is the transcript:

{transcript}

"""

output_parser = StrOutputParser()
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo-preview")
guide_prompt = ChatPromptTemplate.from_template(guide_prompt_template)

guide_chain = (
    {"transcript": RunnablePassthrough()} 
    | guide_prompt
    | llm
    | output_parser
)


url = 'https://www.youtube.com/watch?v=zduSFxRajkE'
path = '/Users/adamlucek/Documents/Jupyter/karpathy_guide_challenge'

print(f"Downloading Video & Audio, Runtime: {show_current_runtime()}")
# download video, audio, and details
av.download_video(url, path)
title = av.get_title(url)
print(f"Video & Audio Downloaded, Runtime: {show_current_runtime()}")

print(f"Checking File Size & Splitting if Necessary, Runtime: {show_current_runtime()}")
# Check filesize, split into multiple files if needed
split_audio_return = False
av.split_audio(f"{path}/original_files/audio/audio_file.mp4")
print(f"Audio Checked & Split, Runtime: {show_current_runtime()}")

print(f"Processing Audio File with Whisper-1, Runtime: {show_current_runtime()}")
# Process audio files with Whisper and create JSON files of output
av.create_json(split_audio_return, path)
print(f"Audio Processed with Whisper-1, Runtime: {show_current_runtime()}")

print(f"Cleaning Data, Runtime: {show_current_runtime()}")
# Combine if needed, clean extra data
av.process_transcripts(f"{path}/transcript_json", split_audio_return)
print(f"Data Cleaned, Runtime: {show_current_runtime()}")

print(f"Pulling Full Transcript, Runtime: {show_current_runtime()}")
# Pull the full transcript
video_transcript = ut.full_transcript(f'{path}/transcript_json/combined_data.json')
print(f"Transcript Pulled, Runtime: {show_current_runtime()}")

print(f"Chunking & Splitting Transcript, Runtime: {show_current_runtime()}")
# Embed and chunk transcript
text_splitter = SemanticChunker(OpenAIEmbeddings())
split_docs = text_splitter.create_documents([video_transcript])
merged_docs = ut.merge_short_documents(split_docs)
print(f"Transcript Chunked, Runtime: {show_current_runtime()}")

print(f"Embedding Transcript, Runtime: {show_current_runtime()}")
# Embed documents
json_loader = JSONLoader(f"{path}/transcript_json/combined_data.json", jq_schema=".[]", text_content=False)
json_texts = json_loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(json_texts, embeddings)
print(f"Transcript Embedded, Runtime: {show_current_runtime()}")

print(f"Generating Markdown Outline with GPT-4-T, Runtime: {show_current_runtime()}")
# Generate markdown of file with GPT-4-T
ut.generate_markdown(merged_docs, path, guide_chain)
print(f"Markdown File Generated, Runtime: {show_current_runtime()}")

print(f"Replacing Placeholders With Pictures & Links, Runtime: {show_current_runtime()}")
# Replace placeholders with hyperlinks and pictures
ut.convert_txt(path, title, db,url)
print(f"Report Finished, Runtime: {show_current_runtime()}")