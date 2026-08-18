from pydantic import BaseModel
from typing import Optional

class SummaryRequest(BaseModel):
    query: str

class SummaryResponse(BaseModel):
    status: str
    video_id: Optional[str] = ""
    video_title: Optional[str] = ""
    video_url: Optional[str] = ""
    summary: Optional[str] = ""
    error: Optional[str] = None