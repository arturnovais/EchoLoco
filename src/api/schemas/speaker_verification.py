from pydantic import BaseModel

class SpeakerResponse(BaseModel):
    speaker: str