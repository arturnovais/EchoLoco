import nemo.collections.asr as nemo_asr

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
    # Extrai embeddings
    emb1 = model.get_embedding(file1)
    emb2 = model.get_embedding(file2)
    print("Embedding arquivo 1:", emb1.shape)
    print("Embedding arquivo 2:", emb2.shape)

    # Verifica se os audios são do mesmo locutor
    result = model.verify_speakers(file1, file2)
    print(f"Mesma pessoa? {result}")
    return result