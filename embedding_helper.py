"""
embedding_helper.py

This file creates embeddings using OpenAI.

An embedding turns text into a list of numbers so ChromaDB
can find document chunks with similar meaning.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


# Load the OPENAI_API_KEY from your .env file
load_dotenv()

# Create OpenAI client using your key from .env
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embedding model used for the project
EMBEDDING_MODEL = "text-embedding-3-small"


def create_embedding(text):
    """
    Send text to OpenAI and return its embedding vector.

    Parameters:
        text: Text that will be converted into an embedding.

    Returns:
        A list of numbers representing the meaning of the text.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding

def transcribe_audio(audio_bytes):
    """
    Convert recorded microphone audio into text using
    OpenAI Whisper.
    """

    # Whisper needs a file-like object with a name.
    import io

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "speech.wav"

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    return transcript.text