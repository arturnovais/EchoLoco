from fastapi import FastAPI
from api.routes import tts, stt, speaker_verification, assistant

app = FastAPI(
    title="EchoLoco API",
    description="API para síntese, transcrição e reconhecimento de voz",
    version="1.0.0"
)

# Registrar as rotas
app.include_router(tts.router, prefix="/tts", tags=["TTS"])
app.include_router(stt.router, prefix="/stt", tags=["STT"])
app.include_router(speaker_verification.router, prefix="/speaker", tags=["Speaker Recognition"])
app.include_router(assistant.router, prefix="/assistant", tags=["Assistant"])