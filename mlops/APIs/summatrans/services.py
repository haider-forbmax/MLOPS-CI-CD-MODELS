###services.py
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from config import Config

settings = Config()

# Initialize OpenAI client (compatible with OpenAI API structure)
client = OpenAI(
    base_url=settings.BASE_URL,
    api_key=settings.LLM_API_KEY
)

# Internal Pydantic models for parsing (structured outputs)
class TranslationOutput(BaseModel):
    translated_text: str = Field(description="The translated text")
    detected_language: str = Field(description="Detected source language (English or Urdu)")

class SummarizationOutput(BaseModel):
    summary: str = Field(description="The generated summary in English")
    detected_language: str = Field(description="Detected language of input text")


class TranslationService:
    """Service for handling English ↔ Urdu translation"""

    @staticmethod
    def translate_text(text: str) -> TranslationOutput:
        """
        Translates text between English and Urdu.
        - If Urdu → translates to English
        - If English → translates to Urdu
        Args:
            text: Text to translate
        Returns:
            TranslationOutput object with translated text and detected language
        """
        system_prompt = f"""You are an expert translator specializing in English and Urdu translation.
Your task:
{Config.DYNAMIC_TRANSLATION_PROMPT}
Examples:
{Config.TRANSLATION_EXAMPLES}
**Translation Guidelines:**
- Maintain the original meaning and tone
- Use natural, fluent language in the target language
- Preserve cultural context appropriately
- Keep proper nouns when appropriate
- Provide accurate detection of the source language
**Important:**
- Only detect "English" or "Urdu" as the language
- Provide high-quality, accurate translations"""

        # CORRECT OpenAI structured output structure (beta.parse)
        response = client.beta.chat.completions.parse(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Translate this text (if Urdu→English, if English→Urdu):\n\n{text}"
                }
            ],
            response_format=TranslationOutput,   # Correct parameter name
        )
        return response.choices[0].message.parsed


class SummarizationService:
    """Service for handling text summarization (always outputs English)"""

    @staticmethod
    def summarize_text(text: str, max_words: int = 150) -> SummarizationOutput:
        """
        Summarizes text in English.
        - Detects if input is English or Urdu
        - Always provides summary in English
        Args:
            text: Text to summarize (English or Urdu)
            max_words: Maximum words in summary
        Returns:
            SummarizationOutput object with English summary and detected language
        """
        system_prompt = f"""You are an expert at creating clear, concise summaries in English.
Your task:
{Config.DYNAMIC_SUMMARIZATION_PROMPT}
Examples:
{Config.SUMMARIZATION_EXAMPLES}
**Summarization Guidelines:**
- Capture the main ideas and key points
- Write in clear, natural English
- Maintain logical flow
- Be objective and accurate
- Stay within the word limit
- If input is Urdu, understand it first then summarize in English
**Output Requirements:**
- Summary must be in English
- Properly formatted and coherent
- Under {max_words} words"""

        # CORRECT OpenAI structured output structure (beta.parse)
        response = client.beta.chat.completions.parse(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Summarize this text in English:\n\n{text}"
                }
            ],
            response_format=SummarizationOutput,   # Correct parameter name
        )
        return response.choices[0].message.parsed