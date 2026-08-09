"""Text chunker for RAG pipeline."""

from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """Split text into chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Chunk a single text."""
        chunks = self.splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "metadata": metadata or {}
            }
            documents.append(doc)

        return documents

    def chunk_transcripts(self, transcripts: List[Dict]) -> List[Dict]:
        """Chunk multiple transcripts."""
        all_chunks = []

        for transcript in transcripts:
            text = transcript.get("full_transcript", "")
            metadata = {
                "source": transcript.get("file_path", ""),
                "language": transcript.get("language", "en"),
                "duration": transcript.get("duration", 0),
                "file_name": transcript.get("file_name", ""),
            }

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks
