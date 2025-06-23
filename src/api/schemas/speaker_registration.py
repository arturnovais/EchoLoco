from pydantic import BaseModel, Field
from typing import Optional

class SpeakerRegisterRequest(BaseModel):
    speaker_name: str = Field(..., description="Nome do locutor que será cadastrado")
    audio_path: str   = Field(..., description="Caminho do áudio (pode ser gs://bucket/obj.wav ou caminho local)")
    speaker_id: Optional[str] = Field(
        None, description="UUID opcional a ser usado como ID no Qdrant"
    )

class SpeakerRegisterResponse(BaseModel):
    speaker_id: str = Field(..., description="UUID gerado/atribuído no Qdrant")
    status: str     = Field(..., description="Mensagem de status da operação")
