from fastapi import APIRouter
from api.schemas.tts import TTSRequest, TTSResponse
from services.tts.tts import tts_from_text, load_tts_pipeline
import uuid

router = APIRouter()

# Carrega modelo na inicialização
tts_tuple, language, output_dir, audio_format = load_tts_pipeline()

@router.post("/", response_model=TTSResponse)
def tts_generate(request: TTSRequest):
    file_id = str(uuid.uuid4())

    gs_path = tts_from_text(
        text=request.text,
        tts_tuple=tts_tuple,
        language=language,
        output_dir=output_dir,  
        file_name=file_id,
        audio_format=audio_format
    )
    return TTSResponse(audio_path=gs_path)  
