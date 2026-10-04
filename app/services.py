import os
import re
import tempfile
from typing import Optional
from urllib.parse import urlparse, parse_qs
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from groq import Groq

# Initialize Groq client using environment variable GROQ_API_KEY
api_key = os.getenv("GROQ_API_KEY", "Your_api_key_goes_here.")
groq_client = Groq(api_key=api_key)


def extract_video_id(url_or_topic: str) -> Optional[str]:
    """Extracts 11-character YouTube video ID if input is a URL."""
    parsed = urlparse(url_or_topic)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("youtube.com", "www.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_topic)
    return match.group(1) if match else None


def get_single_youtube_video(query: str) -> dict:
    """Fetches details for top search result or direct URL target."""
    video_id = extract_video_id(query)
    if video_id:
        return {"video_id": video_id, "title": f"YouTube Video ({video_id})"}

    ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': True,
    'extractor_args': {
        'youtube': {
            'player_client': ['web_safari', 'android']
            }
        }
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        if 'entries' in info and len(info['entries']) > 0:
            top_result = info['entries'][0]
            return {
                "video_id": top_result['id'],
                "title": top_result.get('title', 'YouTube Video')
            }
            
    raise ValueError(f"No YouTube video results found for: {query}")


def fetch_official_captions(video_id: str) -> Optional[str]:
    """Attempts to fetch text captions directly via youtube-transcript-api."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'ur'])
        if hasattr(fetched_transcript, 'to_raw_data'):
            raw_data = fetched_transcript.to_raw_data()
            return " ".join([item['text'] for item in raw_data])
        else:
            return " ".join([item.text for item in fetched_transcript])
    except Exception:
        return None


def fetch_whisper_transcription(video_id: str) -> str:
    """Downloads audio using iOS client impersonation and transcribes via Groq Whisper."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        output_template = os.path.join(temp_dir, "audio.%(ext)s")
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'nopart': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android'],
                }
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
            }
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            downloaded_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
            if not downloaded_files:
                raise ValueError("Could not download audio stream.")
            
            target_file = downloaded_files[0]

            with open(target_file, "rb") as f:
                file_bytes = f.read()

            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(target_file), file_bytes),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
            return transcription

        except Exception as e:
            err_str = str(e)
            if "Sign in to confirm" in err_str or "bot" in err_str:
                raise ValueError("YouTube anti-bot rate limit active on current IP. Try changing IP or passing cookies.")
            raise ValueError(f"Audio transcription failed: {err_str}")


def get_transcript_hybrid(video_id: str) -> str:
    """
    Hybrid Pipeline:
    1. First tries direct captions (fast, no audio download needed).
    2. Falls back to Groq Whisper audio transcription if captions are missing/disabled.
    """
    # 1. Try fetching official text captions
    caption_text = fetch_official_captions(video_id)
    if caption_text and caption_text.strip():
        return caption_text

    # 2. Fall back to Groq Whisper Speech-to-Text
    return fetch_whisper_transcription(video_id)


def summarize_transcript(transcript: str, video_title: str) -> str:
    """Summarizes transcript using Groq Llama 3 with strict grounding rules."""
    system_prompt = (
        "You are an objective AI assistant. Provide a comprehensive summary of the transcript.\n"
        "STRICT GROUNDING RULES:\n"
        "1. Base your summary ONLY on facts mentioned in the transcript.\n"
        "2. Do NOT add outside facts or assumptions.\n"
        "3. Format cleanly with key takeaways and bullet points."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Title: {video_title}\n\nTranscript:\n{transcript}"}
        ],
    )
    return response.choices[0].message.content
