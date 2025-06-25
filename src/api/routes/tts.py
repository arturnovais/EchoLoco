from fastapi import APIRouter, HTTPException
from api.schemas.tts import TTSRequest, TTSResponse, ElevenTTSRequest, ElevenTTSResponse
from services.tts.tts import tts_from_text, load_tts_pipeline, tts_eleven
import uuid

router = APIRouter()

# Carrega modelo na inicialização
tts_tuple, language, output_dir, audio_format = load_tts_pipeline()

@router.post("/pretrained", response_model=TTSResponse)
def tts_generate(request: TTSRequest):
    """
    Converte texto em fala usando modelo pré-treinado local
    """
    try:
        file_id = str(uuid.uuid4())

        gs_path = tts_from_text(
            text=request.text,
            tts_tuple=tts_tuple,
            language=language,
            output_dir=output_dir,  
            file_name=file_id,
            audio_format=audio_format
        )
        return TTSResponse(audio_path=gs_path, file_id=file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/elevenlabs", response_model=ElevenTTSResponse)
def tts_generate_eleven(req: ElevenTTSRequest):
    """
    Converte texto em fala usando ElevenLabs e retorna gs://...
    """
    try:
        gs_uri = tts_eleven(
            text=req.text,
            voice_id=req.voice_id or None,
            voice_settings=req.voice_settings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ElevenTTSResponse(audio_path=gs_uri)
