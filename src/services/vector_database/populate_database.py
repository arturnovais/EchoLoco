import nemo.collections.asr as nemo_asr
import tempfile
import soundfile as sf
import librosa
import pandas as pd
import os
from pathlib import Path
from src.services.vector_database.qdrant_service import QdrantService


def load_model():
    """
    Carrega o modelo pré-treinado de reconhecimento de locutor.
    """
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    print("Modelo carregado.")
    return model


def extract_embedding(model, audio_path):
    """
    Extrai o embedding do arquivo de áudio com Titanet.

    Args:
        model: Modelo Titanet carregado.
        audio_path: Caminho para o arquivo de áudio.

    Returns:
        list[float]: O vetor do embedding.
    """
    # Carrega e resample
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Salva temporariamente como wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name

    # Extrai embedding
    embedding = model.get_embedding(tmp_path)[0].cpu().numpy().tolist()

    # Remove o arquivo temporário
    os.unlink(tmp_path)

    return embedding


def process_audios(audio_dir: Path, metadata_path: Path, qdrant_service):
    df = pd.read_csv(metadata_path)

    model = load_model()
    qdrant_service.create_collection(force_recreate=True)

    for _, row in df.iterrows():
        raw_path = str(row["path"]).strip().lstrip("/")  # remove barra inicial se existir
        rel_path = Path(raw_path)

        # Se o caminho começar com 'audios/', remova para evitar redundância
        if rel_path.parts and rel_path.parts[0] == 'audios':
            rel_path = Path(*rel_path.parts[1:])

        audio_path = audio_dir / rel_path

        if not audio_path.exists():
            print(f"Arquivo não encontrado: {audio_path}")
            continue

        try:
            print(f"Processando: {audio_path}")
            embedding = extract_embedding(model, str(audio_path))

            payload = {
                "path": str(raw_path),
                "ruido": bool(row["ruído?"]),
                "transcricao": row["transcrição"]
            }

            record_id = qdrant_service.insert_embedding(embedding, payload=payload)
            print(f"Embedding inserido com ID: {record_id}")

        except Exception as e:
            print(f"Erro ao processar {audio_path}: {e}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[3]  
    AUDIO_DIR = BASE_DIR / "audios"
    METADATA_CSV = AUDIO_DIR / "audios_metadata.csv"

    qdrant = QdrantService()

    process_audios(AUDIO_DIR, METADATA_CSV, qdrant)