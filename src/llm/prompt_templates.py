"""Prompt templates for multilingual RAG."""

SYSTEM_PROMPT_EN = """You are a helpful assistant that answers questions based on the provided context from audio transcripts.
The context may contain content in English and/or Hindi. Answer in the same language as the user's question.
If the answer is not in the context, say "I don't have enough information to answer that question."
Be concise and accurate. Do not make up information."""

SYSTEM_PROMPT_HI = """आप एक सहायक सहायक हैं जो ऑडियो ट्रांसक्रिप्ट से दिए गए संदर्भ के आधार पर प्रश्नों का उत्तर देते हैं।
संदर्भ में अंग्रेजी और/या हिंदी में सामग्री हो सकती है। उपयोगकर्ता के प्रश्न की भाषा में उत्तर दें।
यदि उत्तर संदर्भ में नहीं है, तो कहें "मुझे उस सवाल का जवाब देने के लिए पर्याप्त जानकारी नहीं है।"
संक्षिप्त और सटीक रहें। जानकारी न बनाएं।"""

QA_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer based on the context above. If the context is not sufficient, say "I don't have enough information to answer that question."
Be concise and accurate. Do not make up information."""

SUMMARIZATION_PROMPT = """Summarize the following transcript concisely. Focus on key points, decisions, and important information.
Preserve the original language of the content.

Transcript:
{transcript}

Summary:"""


def get_system_prompt(language: str = "en") -> str:
    """Get system prompt based on language."""
    return SYSTEM_PROMPT_HI if language == "hi" else SYSTEM_PROMPT_EN


def get_qa_prompt(context: str, question: str) -> str:
    """Get QA prompt with context."""
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def get_summarization_prompt(transcript: str) -> str:
    """Get summarization prompt."""
    return SUMMARIZATION_PROMPT.format(transcript=transcript)
