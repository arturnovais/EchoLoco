from pydantic import BaseModel, Field

class AssistantRequestMessages(BaseModel):
    messages: list[dict] = Field(..., description="Mensagens de entrada fornecidas pelo usuário")

class AssistantRequest(BaseModel):
    user_text: str = Field(..., description="Texto de entrada fornecido pelo usuário")

class AssistantResponse(BaseModel):
    assistant_text: str = Field(..., description="Resposta do assistente")
