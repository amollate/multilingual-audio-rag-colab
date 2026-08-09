"""Transcript processor for Colab environment."""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Processes transcript files for RAG pipeline."""

    def __init__(self):
        self.output_format = "json"

    def load_transcript(self, file_path: str) -> Dict:
        """Load a single transcript file."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("file_name", path.name)
            data.setdefault("file_path", file_path)
            return data
        elif path.suffix.lower() == ".txt":
            return self._parse_line_numbered_text(file_path)
        else:
            raise ValueError(f"Unsupported transcript format: {path.suffix}")

    def _parse_line_numbered_text(self, file_path: str) -> Dict:
        """Parse line-numbered text transcript."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        segments = []
        full_parts = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+):\s*(.+)$", line)
            if match:
                text = match.group(2).strip()
            else:
                text = line
            if text:
                segments.append({"start": 0, "end": 0, "text": text})
                full_parts.append(text)

        full_transcript = " ".join(full_parts)
        language = self._detect_language(full_transcript)

        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "language": language,
            "duration": 0,
            "full_transcript": full_transcript,
            "segments": segments,
        }

    def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not text:
            return "en"
        try:
            from langdetect import detect_langs
            langs = detect_langs(text[:2000])
            if langs and langs[0].lang in ("en", "hi"):
                return langs[0].lang
        except Exception:
            pass
        return "en"

    def load_all_transcripts(self, directory: str) -> List[Dict]:
        """Load all transcripts from directory."""
        transcripts = []
        dir_path = Path(directory)
        for file in sorted(dir_path.iterdir()):
            if file.suffix.lower() in (".json", ".txt"):
                try:
                    transcript = self.load_transcript(str(file))
                    transcript["source_file"] = str(file)
                    transcripts.append(transcript)
                    logger.info(f"Loaded: {file.name} ({transcript.get('language', 'en')}, {len(transcript.get('full_transcript', ''))} chars)")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        return transcripts

    def export_for_rag(self, transcripts: List[Dict], output_path: str) -> List[Dict]:
        """Export transcripts for RAG processing."""
        rag_documents = []
        for t in transcripts:
            doc = {
                "id": Path(t.get("file_path", t.get("source_file", ""))).stem,
                "text": t.get("full_transcript", ""),
                "language": t.get("language", "en"),
                "source": t.get("file_path", t.get("source_file", "")),
                "duration": t.get("duration", 0),
                "metadata": t.get("metadata", {}),
            }
            rag_documents.append(doc)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rag_documents, f, ensure_ascii=False, indent=2)

        return rag_documents
