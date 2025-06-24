from pydantic import BaseModel, Field

class SpeakerRegisterRequest(BaseModel):
    speaker_name: str = Field(..., description="Nome do locutor que será cadastrado")
    audio_path: str   = Field(..., description="Caminho do áudio (pode ser gs://bucket/obj.wav ou caminho local)")
    speaker_id: str = Field(..., description="UUID a ser usado como ID no Qdrant")
    instructions: str = Field(..., description="Instruções para o chatbot")

class SpeakerRegisterResponse(BaseModel):
    speaker_id: str = Field(..., description="UUID gerado/atribuído no Qdrant")
    status: str     = Field(..., description="Mensagem de status da operação")
