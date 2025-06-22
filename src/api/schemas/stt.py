from pydantic import BaseModel

class STTRequest(BaseModel):
    audio_path: str

class STTResponse(BaseModel):
    text: str