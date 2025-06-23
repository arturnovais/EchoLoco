from fastapi import APIRouter
from api.schemas.stt import STTResponse, STTRequest
from services.stt.stt import load_stt_pipeline, stt_from_audio

router = APIRouter()

# Carrega config uma única vez
_provider, _asr_obj, _kwargs = load_stt_pipeline()

@router.post("/", response_model=STTResponse)
def stt_transcribe(request: STTRequest):
    """
    Recebe audio_path (local ou gs://...), chama HuggingFace ou OpenAI.
    """
    text = stt_from_audio(
        request.audio_path,
        provider=_provider,
        asr_obj=_asr_obj,
        transcription_kwargs=_kwargs,
    )
    return STTResponse(text=text)
