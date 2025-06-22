from fastapi import APIRouter
from api.schemas.tts import TTSRequest, TTSResponse
from services.tts.tts import tts_from_text

router = APIRouter()

@router.post("/", response_model=TTSResponse)
def tts_generate(request: TTSRequest):
    audio_path = tts_from_text(request.text)
    return TTSResponse(audio_path=audio_path)