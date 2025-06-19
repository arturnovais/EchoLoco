import nemo.collections.asr as nemo_asr

def speaker_recognition(file1, file2):
    # Carrega modelo pré-treinado
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    print("Modelo carregado:", model)

    # Extrai embeddings
    emb1 = model.get_embedding(file1)
    emb2 = model.get_embedding(file2)
    print("Embedding arquivo 1:", emb1.shape)
    print("Embedding arquivo 2:", emb2.shape)

    # Verifica se os audios são do mesmo locutor
    result = model.verify_speakers(file1, file2)
    print(f"Mesma pessoa? {result}")

    # Exemplo: comparar vários arquivos em loop
    arquivos = [file1, file2]
    embeddings = {f: model.get_embedding(f) for f in arquivos}
    for a in arquivos:
        for b in arquivos:
            score = model.verify_speakers(a, b)
            print(f"{a} vs {b}: {score}")