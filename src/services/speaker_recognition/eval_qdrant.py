import os
import itertools
import argparse
from collections import Counter
import random
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from src.services.vector_database.qdrant_service import QdrantService

# --------------------------------------------------------------------------- #
# 1. Coleta dos arquivos .wav                                                 #
# --------------------------------------------------------------------------- #
def collect_audio_files(base_dir: str) -> Dict[str, List[str]]:
    """Percorre `base_dir` e devolve {locutor: [lista de paths .wav]}."""
    speakers: Dict[str, List[str]] = {}
    for spk in os.listdir(base_dir):
        spk_path = Path(base_dir, spk)
        if spk_path.is_dir():
            wavs = sorted(
                str(spk_path / f)
                for f in os.listdir(spk_path)
                if f.lower().endswith(".wav")
            )
            if wavs:
                speakers[spk] = wavs
    return speakers


# --------------------------------------------------------------------------- #
# 2. Busca de embeddings no Qdrant                                            #
# --------------------------------------------------------------------------- #
def fetch_embedding(qs: QdrantService, wav_path: str) -> Tuple[str, np.ndarray]:
    """
    Retorna (wav_path, embedding) buscando pelo payload 'path' == wav_path.
    Lança ValueError se não encontrar.
    """
    registros = qs.query_by_payload("path", wav_path, limit=1)
    if not registros:
        raise ValueError(f"❌ Embedding não encontrado para {wav_path!r}")
    vec = np.asarray(registros[0].vector, dtype=np.float32)
    return wav_path, vec


def load_all_embeddings(
    qs: QdrantService, speakers: Dict[str, List[str]], threads: int = 8
) -> Dict[str, np.ndarray]:
    """Busca todos os embeddings em paralelo (I/O bound → thread pool)."""
    wav_paths = list(itertools.chain.from_iterable(speakers.values()))
    emb_dict: Dict[str, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for wav, emb in pool.map(lambda p: fetch_embedding(qs, p), wav_paths):
            emb_dict[wav] = emb

    return emb_dict


# --------------------------------------------------------------------------- #
# 3. Geração dos pares                                                        #
# --------------------------------------------------------------------------- #
def generate_pairs(
    speakers: Dict[str, List[str]],
    pairs_per_speaker: int,
    neg_pairs_per_combo: int,
) -> Tuple[List[Tuple[str, str, bool]], List[Tuple[str, str]]]:
    """
    all_info  = [(f1, f2, expected_bool), ...]
    just_pairs = [(f1, f2), ...]
    """
    all_info: List[Tuple[str, str, bool]] = []

    # --- positivos (mesmo locutor) ---
    for files in speakers.values():
        combos = list(itertools.combinations(files, 2))
        sampled = random.sample(combos, min(pairs_per_speaker, len(combos)))
        all_info.extend((f1, f2, True) for f1, f2 in sampled)

    # --- negativos (locutores diferentes) ---
    for spk1, spk2 in itertools.combinations(speakers.keys(), 2):
        files1, files2 = speakers[spk1], speakers[spk2]
        all_neg = [(f1, f2) for f1 in files1 for f2 in files2]
        sampled = random.sample(all_neg, min(neg_pairs_per_combo, len(all_neg)))
        all_info.extend((f1, f2, False) for f1, f2 in sampled)

    just_pairs = [(f1, f2) for f1, f2, _ in all_info]
    return all_info, just_pairs


# --------------------------------------------------------------------------- #
# 4. Verificação usando cosseno                                               #
# --------------------------------------------------------------------------- #
def cosine_same_speaker(
    emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.55
) -> bool:
    cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return cos >= threshold


# --------------------------------------------------------------------------- #
# 5. Avaliação e métricas                                                     #
# --------------------------------------------------------------------------- #
def evaluate(
    all_info: List[Tuple[str, str, bool]],
    embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> None:
    """Calcula predições, imprime métricas e erros."""
    y_true, y_pred = [], []
    for f1, f2, expected in all_info:
        pred = cosine_same_speaker(embeddings[f1], embeddings[f2], threshold)
        y_true.append(expected)
        y_pred.append(pred)

    total = len(y_true)
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    accuracy = correct / total if total else 0.0

    print(f"\nTotal de pares avaliados : {total}")
    print(f"Corretos                 : {correct}")
    print(f"Accuracy                 : {accuracy:.4f}\n")

    counts = Counter(zip(y_true, y_pred))
    print("Matriz de confusão (T=True mesmo locutor):")
    print(f"TP = {counts[(True, True)]} | FN = {counts[(True, False)]}")
    print(f"FP = {counts[(False, True)]} | TN = {counts[(False, False)]}\n")

    print("❌ Comparações incorretas:")
    for (f1, f2, exp), pred in zip(all_info, y_pred):
        if exp != pred:
            print(f"  {f1} <--> {f2} | esperado={exp}  predito={pred}")


# --------------------------------------------------------------------------- #
# 6. Programa principal                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia verificação de locutor usando embeddings "
        "pré-calculados armazenados no Qdrant."
    )
    parser.add_argument(
        "--data_dir",
        default="audios",
        help="Diretório com subpastas por locutor contendo arquivos .wav.",
    )
    parser.add_argument(
        "--pairs_per_speaker",
        type=int,
        default=70,
        help="Quantos pares POSITIVOS amostrar por locutor (padrão = 2).",
    )
    parser.add_argument(
        "--neg_pairs_per_combo",
        type=int,
        default=70,
        help="Quantos pares NEGATIVOS amostrar por par de locutores (padrão = 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Limiar de similaridade de cosseno para 'mesmo locutor'.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Threads para buscar embeddings no Qdrant (padrão = 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # --- coleta arquivos ---
    speakers = collect_audio_files(args.data_dir)
    if len(speakers) < 2:
        raise SystemExit("⚠️  É necessário ter pelo menos dois locutores com áudios.")
    total_wavs = sum(map(len, speakers.values()))
    print(f"Locutores: {len(speakers)} | Áudios: {total_wavs}")

    # --- busca embeddings ---
    print("⏳ Buscando embeddings no Qdrant…")
    qs = QdrantService()
    embeddings = load_all_embeddings(qs, speakers, threads=args.threads)
    print("✅ Embeddings carregados:", len(embeddings))

    # --- gera pares ---
    all_info, _ = generate_pairs(
        speakers,
        pairs_per_speaker=args.pairs_per_speaker,
        neg_pairs_per_combo=args.neg_pairs_per_combo,
    )
    print(f"Pares selecionados para avaliação: {len(all_info)}")

    # --- avaliação ---
    evaluate(all_info, embeddings, args.threshold)


if __name__ == "__main__":
    main()