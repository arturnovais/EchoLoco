import os
import tempfile
from fastapi import APIRouter, HTTPException

from api.schemas.speaker_registration import (
    SpeakerRegisterRequest,
    SpeakerRegisterResponse,
)

from services.vector_database.qdrant_service import QdrantService
from services.speaker_recognition.speaker_recognition import (
    load_model,
    extract_embedding,
)

from infra.storage import gcs_client

router = APIRouter()

_titanet_model = load_model()
_qdrant = QdrantService()
_qdrant.create_collection()

def _get_local_audio_path(audio_path: str) -> str:
    """
    Se `audio_path` for gs://..., baixa para tempfile e devolve esse caminho.
    Caso contrário, devolve o próprio `audio_path`.
    Retorna (tmp_path|audio_path)  e boolean flag se é temporário.
    """
    if audio_path.startswith("gs://"):
        audio_bytes = gcs_client.download_bytes(audio_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name, True
    return audio_path, False

@router.post("/", response_model=SpeakerRegisterResponse)
def register_speaker(req: SpeakerRegisterRequest):
    """
    Cadastra o locutor: extrai embedding, insere no Qdrant e devolve o UUID.
    """
    local_path, is_tmp = _get_local_audio_path(req.audio_path)

    try:
        embedding_vec = extract_embedding(_titanet_model, local_path)
    except Exception as e:
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)
        raise HTTPException(400, f"Erro ao extrair embedding: {e}")

    payload = {
        "speaker_name": req.speaker_name,
        "audio_path": req.audio_path,   
    }

    try:
        speaker_uuid = _qdrant.insert_embedding(
            embedding=embedding_vec,
            record_id=req.speaker_id,
            payload=payload,
            collection_name="speakers",
        )
    except Exception as e:
        raise HTTPException(500, f"Erro ao inserir no Qdrant: {e}")
    finally:
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)

    return SpeakerRegisterResponse(
        speaker_id=speaker_uuid,
        status="registered",
    )
