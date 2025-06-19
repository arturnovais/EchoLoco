import nemo.collections.asr as nemo_asr
import tempfile
import soundfile as sf
import librosa
import os
import numpy as np
import torch

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
    result = model.verify_speakers(temp1.name, temp2.name, threshold=0.8)
    print(f"Mesma pessoa? {result}")
    
    # Remove arquivos temporários
    os.unlink(temp1.name)
    os.unlink(temp2.name)
    
    return result

def _to_numpy(vec):
    """
    Converte torch.Tensor → np.ndarray (float32) e achata dimensão [1, D] se existir.
    Aceita np.ndarray sem alterações.
    """
    if torch is not None and isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    vec = np.asarray(vec, dtype=np.float32)
    return vec.squeeze()  
    
def verify_speakers_cossine(model, file1, file2, threshold=0.6):
    """
    Verifica se dois arquivos de áudio são do mesmo locutor
    """
    audio1, _ = librosa.load(file1, sr=16000, mono=True)
    audio2, _ = librosa.load(file2, sr=16000, mono=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t1, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t2:

        sf.write(t1.name, audio1, 16000)
        sf.write(t2.name, audio2, 16000)

        emb1 = _to_numpy(model.get_embedding(t1.name))
        emb2 = _to_numpy(model.get_embedding(t2.name))
        print("Embedding arquivo 1:", emb1.shape)
        print("Embedding arquivo 2:", emb2.shape)

    os.unlink(t1.name)
    os.unlink(t2.name)

    # --------------------------------------------------------------------- #
    # Similaridade de cosseno                                               #
    # --------------------------------------------------------------------- #
    emb1 = emb1.astype(np.float32)
    emb2 = emb2.astype(np.float32)

    cosine = float(np.dot(emb1, emb2) /
                   (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    print(f"Similaridade de cosseno: {cosine:.4f}  |  "
          f"Limiar: {threshold}")

    result = cosine >= threshold
    print(f"Mesma pessoa? {result}")

    return result

if __name__ == "__main__":
    model = load_model()
    result = verify_speakers_cossine(model, "audios/Gustavo/6_cr.wav", "audios/Artur/5_cr.wav")
    print(result)