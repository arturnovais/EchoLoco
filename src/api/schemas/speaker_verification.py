from pydantic import BaseModel, Field
from typing import Optional

class SpeakerVerificationRequest(BaseModel):
    audio_path: str = Field(
        ...,
        description="Caminho do áudio (gs://bucket/obj.wav ou caminho local)"
    )
    threshold: Optional[float] = Field(
        0.45,
        description="Threshold para verificação do speaker (padrão: 0.45)",
        ge=0.0,
        le=1.0
    )

class SpeakerVerificationResponse(BaseModel):
    matched: bool             = Field(..., description="Se o locutor foi reconhecido")
    speaker_id: Optional[str] = Field(
        None, description="UUID retornado do Qdrant se matched=True"
    )
    score: Optional[float]    = Field(
        None, description="Score/distância devolvido pelo Qdrant"
    )
