"""
chunking.py

This file breaks the Pollution Prevention AI document
into smaller chunks.

The RAG system searches these smaller chunks instead
of searching the whole document at once.
"""


def chunk_text(text, chunk_size=500):
    """
    Split text into chunks.

    Parameters:
        text: The full pollution prevention document.
        chunk_size: Approximately how many characters
                    should be in each chunk.

    Returns:
        A list of cleaned text chunks.
    """

    chunks = []

    # Move through the document chunk_size characters at a time.
    for start in range(0, len(text), chunk_size):

        chunk = text[start:start + chunk_size]

        # Do not add empty chunks.
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks