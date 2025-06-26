"""
Verifica locutor: recebe um áudio, extrai embedding, consulta Qdrant
e devolve o speaker_id se a similaridade (ou distância) ultrapassar o threshold.
"""
import os
import tempfile
import logging
import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.speaker_verification import (
    SpeakerVerificationRequest,
    SpeakerVerificationResponse,
)

from services.vector_database.qdrant_service import QdrantService
from services.speaker_recognition.speaker_recognition import (
    load_model,
    extract_embedding,
)

from infra.storage import gcs_client

# Configura o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

_titanet = load_model()
_qdrant = QdrantService()

def _materialize_audio(path: str) -> tuple[str, bool]:
    """
    Faz download se gs://..., devolve (local_path, is_tmp)
    """
    if path.startswith("gs://"):
        logger.info(f"Downloading audio from {path}")
        data = gcs_client.download_bytes(path)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        logger.info(f"Audio downloaded to temporary file {tmp.name}")
        return tmp.name, True
    return path, False


@router.post("/", response_model=SpeakerVerificationResponse)
def verify_speaker(req: SpeakerVerificationRequest):
    logger.info(f"Received request to verify speaker with audio path: {req.audio_path}")
    logger.info(f"Using threshold: {req.threshold}")
    local_path, is_tmp = _materialize_audio(req.audio_path)

    try:
        logger.info(f"Extracting embedding from audio at {local_path}")
        emb = extract_embedding(_titanet, local_path)
        logger.info("Embedding extracted successfully")
    except Exception as e:
        logger.error(f"Failed to extract embedding: {e}")
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)
        raise HTTPException(400, f"Falha ao extrair embedding: {e}")

    try:
        logger.info("Searching for similar embeddings in Qdrant")
        results = _qdrant.search_similar(embedding=emb, top_k=1, collection_name="speakers", with_vectors=True)
        logger.info("Search completed")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Error during Qdrant search: {e}")
        raise HTTPException(500, f"Erro na busca Qdrant: {e}")
    finally:
        if is_tmp and os.path.exists(local_path):
            os.unlink(local_path)

    if not results:
        logger.info("No similar speakers found")
        return SpeakerVerificationResponse(matched=False, speaker_id=None, score=None)

    best = results[0]   
    found_embedding = best.vector   
    logger.info(f"Best match found with distance: {best.score}")
    
    # Calcula similaridade de cosseno entre os embeddings
    cosine_similarity = float(np.dot(emb, found_embedding) / 
                             (np.linalg.norm(emb) * np.linalg.norm(found_embedding)))
    
    logger.info(f"Cosine similarity: {cosine_similarity:.4f} | Threshold: {req.threshold}")

    if cosine_similarity >= req.threshold:
        logger.info(f"Speaker verified with ID: {best.id}")
        return SpeakerVerificationResponse(
            matched=True,
            speaker_id=str(best.id),
            score=cosine_similarity,
        )
    else:
        logger.info("Speaker not verified, cosine similarity below threshold")
        return SpeakerVerificationResponse(
            matched=False,
            speaker_id=None,
            score=cosine_similarity,
        )