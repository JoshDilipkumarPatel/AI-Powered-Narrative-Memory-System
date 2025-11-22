from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import uuid

class MemoryObject(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content_summary: str  

    characters: List[str]
    setting: Optional[str] = None
    key_events: List[str]
    themes: List[str]

    importance_score: float = Field(ge=0.0, le=1.0)  
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 1  

    embedding: List[float]  