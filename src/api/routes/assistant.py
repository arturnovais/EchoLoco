from fastapi import APIRouter
from api.schemas.assistant import AssistantRequest, AssistantResponse, AssistantRequestMessages
from services.assistant.assistant import Assistant

router = APIRouter()

# Carrega o assistente na inicialização
assistant = Assistant()

@router.post("/reply", response_model=AssistantResponse)
def get_assistant_reply(request: AssistantRequest):
    """
    Gera uma resposta do assistente com base no texto do usuário.
    """
    reply_text = assistant.reply(request.user_text)
    return AssistantResponse(assistant_text=reply_text)

@router.post("/reply_api", response_model=AssistantResponse)
def get_assistant_reply_api(request: AssistantRequestMessages):
    """
    Gera uma resposta do assistente com base no texto do usuário.
    """
    reply_text = assistant.reply_api(request.messages)
    return AssistantResponse(assistant_text=reply_text)

@router.get("/welcome", response_model=AssistantResponse)
def assistant_welcome():
    """
    Retorna a mensagem de boas-vindas do assistente.
    """
    return AssistantResponse(assistant_text=assistant.welcome())

@router.get("/goodbye", response_model=AssistantResponse)
def assistant_goodbye():
    """
    Retorna a mensagem de despedida do assistente.
    """
    return AssistantResponse(assistant_text=assistant.goodbye())
