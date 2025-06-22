from fastapi import APIRouter, UploadFile, File
from api.schemas.stt import STTResponse, STTRequest
from services.stt.stt import stt_from_audio

router = APIRouter()

@router.post("/", response_model=STTResponse)
def stt_transcribe(request: STTRequest):
    text = stt_from_audio(request.audio_path)
    return STTResponse(text=text)
