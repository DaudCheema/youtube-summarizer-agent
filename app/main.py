from fastapi import FastAPI, HTTPException
from app.schemas import SummaryRequest, SummaryResponse
from app.services import (
    get_single_youtube_video,
    get_transcript_hybrid,
    summarize_transcript
)

app = FastAPI(title="YouTube Video Summarizer Agent")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_video(payload: SummaryRequest):
    query = payload.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    v_id = ""
    v_title = ""
    v_url = ""

    try:
        # Step 1: Resolve metadata (direct URL or top search topic)
        video_meta = get_single_youtube_video(query)
        v_id = video_meta["video_id"]
        v_title = video_meta["title"]
        v_url = f"https://www.youtube.com/watch?v={v_id}"

        # Step 2: Extract Transcript (Hybrid method)
        transcript = get_transcript_hybrid(v_id)

        # Step 3: Generate unhallucinated summary via LLM
        summary = summarize_transcript(transcript=transcript, video_title=v_title)

        return SummaryResponse(
            status="success",
            video_id=v_id,
            video_title=v_title,
            video_url=v_url,
            summary=summary
        )

    except ValueError as ve:
        return SummaryResponse(
            status="error",
            video_id=v_id,
            video_title=v_title,
            video_url=v_url,
            summary="",
            error=str(ve)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")