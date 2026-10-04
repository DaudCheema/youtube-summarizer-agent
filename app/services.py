import os
import re
import tempfile
from typing import Optional
from urllib.parse import urlparse, parse_qs
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq

api_key = os.getenv("GROQ_API_KEY", "")
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
            parts = parsed.path.split("/")
            if len(parts) > 2:
                return parts[2]
    
    match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url_or_topic)
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
                'player_client': ['android', 'ios']
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


import requests

def fetch_official_captions(video_id: str) -> Optional[str]:
    """Fetches manual or auto-generated captions, falling back to public Invidious proxies."""
    # Attempt 1: Standard youtube-transcript-api
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        if transcript_list:
            return " ".join([item['text'] for item in transcript_list])
    except Exception:
        pass

    # Attempt 2: Invidious Proxy (Bypasses Datacenter IP Bot Challenge)
    instances = [
        "https://inv.tux.pizza",
        "https://invidious.nerdvpn.de",
        "https://vid.puffyan.us"
    ]
    for inst in instances:
        try:
            res = requests.get(f"{inst}/api/v1/captions/{video_id}", timeout=5)
            if res.status_code == 200:
                captions = res.json().get("captions", [])
                if captions:
                    caption_url = inst + captions[0].get("url")
                    cap_res = requests.get(caption_url, timeout=5)
                    # Clean out basic XML/VTT tags
                    clean_text = re.sub(r'<[^>]+>', ' ', cap_res.text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    if len(clean_text) > 50:
                        return clean_text
        except Exception:
            continue

    return None


def fetch_whisper_transcription(video_id: str) -> str:
    """Downloads audio using Android client spoofing and transcribes via Groq Whisper."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        output_template = os.path.join(temp_dir, "audio.%(ext)s")
        
        ydl_opts = {
            'format': 'ba/b',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'nopart': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android'],
                    'player_skip': ['webpage', 'configs']
                }
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
            }
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            downloaded_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
            if not downloaded_files:
                raise ValueError("Could not extract audio track.")
            
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
            if "bot" in err_str.lower() or "sign in" in err_str.lower():
                raise ValueError(
                    "Cloud Datacenter Rate Limit: YouTube blocked media stream extraction from this IP subnet. "
                    "Please test using a video with captions enabled or run locally with local IP."
                )
            raise ValueError(f"Transcription failed: {err_str}")


def get_transcript_hybrid(video_id: str) -> str:
    """
    Hybrid Pipeline:
    1. Direct transcript scraping (bypasses media streaming bot checks).
    2. Fallback to Groq Whisper Speech-to-Text when captions are absent.
    """
    caption_text = fetch_official_captions(video_id)
    if caption_text and caption_text.strip():
        return caption_text

    return fetch_whisper_transcription(video_id)


def summarize_transcript(transcript: str, video_title: str) -> str:
    """Summarizes transcript using Groq Llama 3.3 70B with strict grounding rules."""
    system_prompt = (
        "You are BriefTube AI, an expert research intelligence assistant. Provide a comprehensive summary of the transcript.\n"
        "STRICT GROUNDING RULES:\n"
        "1. Base your summary ONLY on facts mentioned in the transcript.\n"
        "2. Do NOT add outside facts, speculative claims, or assumptions.\n"
        "3. Format cleanly with: Executive Summary, Key Highlights / Takeaways, and Critical Entities/Dates."
    )

    # Protect context length limits
    truncated_transcript = transcript[:15000]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Title: {video_title}\n\nTranscript:\n{truncated_transcript}"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
