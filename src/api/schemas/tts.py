from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_path: str

class ElevenTTSRequest(BaseModel):
    text: str = Field(..., description="Texto a ser sintetizado")
    voice_id: Optional[str] = Field(
        None, description="Voice ID da ElevenLabs (opcional; usa default se omitido)"
    )
    voice_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Dict com 'stability', 'similarity_boost' etc. "
        "(deixa em branco para usar defaults)",
    )

class ElevenTTSResponse(BaseModel):
    audio_path: str = Field(..., description="Caminho gs:// do áudio gerado")