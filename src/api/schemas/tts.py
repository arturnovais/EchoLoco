from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_path: str
