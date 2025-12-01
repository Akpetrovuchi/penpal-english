"""
Voice message processing module for PenPal English Bot.

Functions:
- download_voice_file: Download voice file from Telegram
- transcribe_audio: Transcribe audio using OpenAI Whisper
- analyze_voice_text: Analyze transcribed text for grammar/vocabulary
"""

import os
import logging
import tempfile
import openai
import json
from typing import Optional, Tuple
from aiogram import Bot, types


async def download_voice_file(bot: Bot, message: types.Message) -> Tuple[Optional[str], Optional[str]]:
    """
    Download voice file from Telegram message.
    
    Returns:
        Tuple of (file_path, error_message)
        - file_path: Path to downloaded .ogg file or None if failed
        - error_message: Error description or None if success
    """
    try:
        voice = message.voice
        if not voice:
            return None, "No voice message found"
        
        file_id = voice.file_id
        duration = voice.duration
        
        # Check duration limit (max 2 minutes for reasonable processing)
        if duration > 120:
            return None, "Voice message too long (max 2 minutes)"
        
        # Get file info from Telegram
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path
        
        # Create temp file for download
        # Telegram sends voice as .ogg (Opus codec)
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, f"voice_{message.from_user.id}_{message.message_id}.ogg")
        
        # Download file
        await bot.download_file(file_path, local_path)
        
        logging.info(f"[voice] Downloaded voice file: {local_path}, duration={duration}s, size={voice.file_size}")
        
        return local_path, None
        
    except Exception as e:
        logging.exception(f"[voice] Failed to download voice file: {e}")
        return None, f"Download failed: {str(e)}"


async def transcribe_audio(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        file_path: Path to audio file (.ogg, .mp3, .wav, etc.)
    
    Returns:
        Tuple of (transcribed_text, error_message)
    """
    try:
        # OpenAI Whisper supports: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg
        # Telegram voice messages are .ogg (Opus), which is supported directly
        
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="en"  # Expect English for language learning
            )
        
        text = response.get("text", "").strip()
        
        if not text:
            return None, "Could not recognize any speech"
        
        logging.info(f"[voice] Transcribed: '{text[:100]}...' ({len(text)} chars)")
        
        return text, None
        
    except openai.error.OpenAIError as e:
        logging.exception(f"[voice] Whisper API error: {e}")
        return None, f"Transcription failed: {str(e)}"
    except Exception as e:
        logging.exception(f"[voice] Transcription error: {e}")
        return None, f"Transcription failed: {str(e)}"


async def analyze_voice_text(text: str, user_level: str = "B1") -> dict:
    """
    Analyze transcribed text for grammar and vocabulary.
    
    Args:
        text: Transcribed text from voice message
        user_level: User's English level (A1-C2)
    
    Returns:
        Dictionary with analysis results:
        {
            "original": str,
            "corrected": str,
            "has_errors": bool,
            "grammar_feedback": str,
            "vocabulary_feedback": str,
            "pronunciation_tips": str,
            "score": int (0-100)
        }
    """
    try:
        prompt = f"""You are Max, a friendly English tutor helping a {user_level} level student.

Analyze this spoken English text and provide feedback:
"{text}"

Respond in JSON format:
{{
    "original": "<the original text>",
    "corrected": "<corrected version if there are errors, or same as original if perfect>",
    "has_errors": <true if there are grammar/vocabulary errors, false if text is correct>,
    "grammar_feedback": "<brief, friendly explanation of grammar issues in Russian, or '✨ Грамматика отличная!' if no errors>",
    "vocabulary_feedback": "<brief vocabulary tips or alternatives in Russian, or '👍 Хороший выбор слов!' if good>",
    "pronunciation_tips": "<1-2 words that might be tricky to pronounce, with phonetic hint, or empty if none>",
    "score": <0-100 overall score>
}}

Keep feedback SHORT and FRIENDLY (Max's style - supportive, not lecturing).
Focus on 1-2 most important issues, not everything.
If the text is good, celebrate it!"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        # Ensure all required fields
        result.setdefault("original", text)
        result.setdefault("corrected", text)
        result.setdefault("has_errors", False)
        result.setdefault("grammar_feedback", "")
        result.setdefault("vocabulary_feedback", "")
        result.setdefault("pronunciation_tips", "")
        result.setdefault("score", 80)
        
        logging.info(f"[voice] Analysis complete: score={result['score']}, has_errors={result['has_errors']}")
        
        return result
        
    except json.JSONDecodeError as e:
        logging.exception(f"[voice] Failed to parse GPT response as JSON: {e}")
        return {
            "original": text,
            "corrected": text,
            "has_errors": False,
            "grammar_feedback": "Не удалось проанализировать текст",
            "vocabulary_feedback": "",
            "pronunciation_tips": "",
            "score": 0
        }
    except Exception as e:
        logging.exception(f"[voice] Analysis error: {e}")
        return {
            "original": text,
            "corrected": text,
            "has_errors": False,
            "grammar_feedback": f"Ошибка анализа: {str(e)}",
            "vocabulary_feedback": "",
            "pronunciation_tips": "",
            "score": 0
        }


def format_voice_feedback(analysis: dict) -> str:
    """
    Format analysis results as a user-friendly message.
    
    Args:
        analysis: Dictionary from analyze_voice_text()
    
    Returns:
        Formatted message string
    """
    original = analysis.get("original", "")
    corrected = analysis.get("corrected", "")
    has_errors = analysis.get("has_errors", False)
    grammar_feedback = analysis.get("grammar_feedback", "")
    vocabulary_feedback = analysis.get("vocabulary_feedback", "")
    pronunciation_tips = analysis.get("pronunciation_tips", "")
    score = analysis.get("score", 0)
    
    # Score emoji
    if score >= 90:
        score_emoji = "🌟"
    elif score >= 70:
        score_emoji = "👍"
    elif score >= 50:
        score_emoji = "💪"
    else:
        score_emoji = "📚"
    
    lines = [
        "🎤 <b>Твоё голосовое сообщение:</b>",
        "",
        f"📝 <i>{original}</i>",
    ]
    
    if has_errors and corrected != original:
        lines.extend([
            "",
            f"✨ <b>Исправленный вариант:</b>",
            f"<i>{corrected}</i>",
        ])
    
    if grammar_feedback:
        lines.extend([
            "",
            f"📖 <b>Грамматика:</b> {grammar_feedback}",
        ])
    
    if vocabulary_feedback:
        lines.extend([
            f"💬 <b>Лексика:</b> {vocabulary_feedback}",
        ])
    
    if pronunciation_tips:
        lines.extend([
            f"🗣 <b>Произношение:</b> {pronunciation_tips}",
        ])
    
    lines.extend([
        "",
        f"{score_emoji} <b>Оценка:</b> {score}/100",
    ])
    
    return "\n".join(lines)


def format_text_feedback(analysis: dict) -> str:
    """
    Format analysis results for TEXT messages (not voice).
    Similar to voice feedback but without original text echo and pronunciation tips.
    
    Args:
        analysis: Dictionary from analyze_voice_text()
    
    Returns:
        Formatted message string, or empty string if no errors
    """
    corrected = analysis.get("corrected", "")
    original = analysis.get("original", "")
    has_errors = analysis.get("has_errors", False)
    grammar_feedback = analysis.get("grammar_feedback", "")
    vocabulary_feedback = analysis.get("vocabulary_feedback", "")
    score = analysis.get("score", 0)
    
    # If no errors, return empty - no need for feedback card
    if not has_errors and score >= 90:
        return ""
    
    # Score emoji
    if score >= 90:
        score_emoji = "🌟"
    elif score >= 70:
        score_emoji = "👍"
    elif score >= 50:
        score_emoji = "💪"
    else:
        score_emoji = "📚"
    
    lines = []
    
    if has_errors and corrected != original:
        lines.extend([
            f"✨ <b>Исправленный вариант:</b>",
            f"<i>{corrected}</i>",
            "",
        ])
    
    if grammar_feedback and grammar_feedback not in ["✨ Грамматика отличная!", "✨ Грамматика отличная"]:
        lines.append(f"📖 <b>Грамматика:</b> {grammar_feedback}")
    
    if vocabulary_feedback and vocabulary_feedback not in ["👍 Хороший выбор слов!", "👍 Хороший выбор слов"]:
        lines.append(f"💬 <b>Лексика:</b> {vocabulary_feedback}")
    
    if lines:
        lines.append("")
        lines.append(f"{score_emoji} <b>Оценка:</b> {score}/100")
    
    return "\n".join(lines)


def cleanup_voice_file(file_path: str) -> None:
    """Remove temporary voice file after processing."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"[voice] Cleaned up temp file: {file_path}")
    except Exception as e:
        logging.warning(f"[voice] Failed to cleanup temp file: {e}")
