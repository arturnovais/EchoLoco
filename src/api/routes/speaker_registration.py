import os
import tempfile
import logging
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
from infra.bq.bq_client import insert_rows

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"Downloading audio from GCS: {audio_path}")
        audio_bytes = gcs_client.download_bytes(audio_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        logger.info(f"Audio downloaded to temporary file: {tmp.name}")
        return tmp.name, True
    logger.info(f"Using local audio path: {audio_path}")
    return audio_path, False

@router.post("/", response_model=SpeakerRegisterResponse)
def register_speaker(req: SpeakerRegisterRequest):
    """
    Cadastra o locutor: extrai embedding, insere no Qdrant e devolve o UUID.
    """
    logger.info(f"Registering speaker: {req.speaker_name}")
    local_path, is_tmp = _get_local_audio_path(req.audio_path)

    try:
        logger.info("Extracting embedding from audio")
        embedding_vec = extract_embedding(_titanet_model, local_path)
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)
        raise HTTPException(400, f"Erro ao extrair embedding: {e}")

    payload = {
        "speaker_name": req.speaker_name,
        "audio_path": req.audio_path,   
        "speaker_id": req.speaker_id
    }

    try:
        logger.info("Inserting embedding into Qdrant")
        speaker_uuid = _qdrant.insert_embedding(
            embedding=embedding_vec,
            record_id=req.speaker_id,
            payload=payload,
            collection_name="speakers",
        )
    except Exception as e:
        logger.error(f"Error inserting into Qdrant: {e}")
        raise HTTPException(500, f"Erro ao inserir no Qdrant: {e}")
    finally:
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)
    
    bq_row = {
        "speaker_id": req.speaker_id,
        "speaker_name": req.speaker_name,
        "instructions": req.instructions,
    }
    logger.info("Inserting speaker data into BigQuery")
    bq_errors = insert_rows("system_prompts", [bq_row])

    if bq_errors:
        logger.error(f"Error inserting into BigQuery: {bq_errors}")
        raise HTTPException(500, f"Erro ao inserir no BigQuery: {bq_errors}")

    logger.info(f"Speaker registered successfully: {speaker_uuid}")
    return SpeakerRegisterResponse(
        speaker_id=speaker_uuid,
        status="registered"
    )
