import nemo.collections.asr as nemo_asr
import tempfile
import soundfile as sf
import librosa
import os

def load_model():
    """
    Carrega o modelo pré-treinado de reconhecimento de locutor
    """
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    print("Modelo carregado:", model)
    return model

def verify_speakers(model, file1, file2):
    """
    Verifica se dois arquivos de áudio são do mesmo locutor
    
    Args:
        model: Modelo pré-treinado de reconhecimento de locutor
        file1: Caminho para o primeiro arquivo de áudio
        file2: Caminho para o segundo arquivo de áudio
        
    Returns:
        bool: True se os áudios são do mesmo locutor, False caso contrário
    """
    # Carrega e resample os áudios para 16kHz mono
    audio1, _ = librosa.load(file1, sr=16000, mono=True)
    audio2, _ = librosa.load(file2, sr=16000, mono=True)
    
    temp1 = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp2 = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    sf.write(temp1.name, audio1, 16000)
    sf.write(temp2.name, audio2, 16000)

    # Extrai embeddings
    emb1 = model.get_embedding(temp1.name)
    emb2 = model.get_embedding(temp2.name)
    print("Embedding arquivo 1:", emb1.shape)
    print("Embedding arquivo 2:", emb2.shape)

    # Verifica se os audios são do mesmo locutor
    result = model.verify_speakers(temp1.name, temp2.name)
    print(f"Mesma pessoa? {result}")
    
    # Remove arquivos temporários
    os.unlink(temp1.name)
    os.unlink(temp2.name)
    
    return result

if __name__ == "__main__":
    model = load_model()
    result = verify_speakers(model, "audios/Gustavo/4_sr.wav", "audios/Artur/1_sr.wav")
    print(result)