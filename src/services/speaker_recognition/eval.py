import os
import itertools
import argparse
from collections import Counter
import random
from speaker_recognition import load_model, verify_speakers
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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
# 2. Geração dos pares                                                        #
# --------------------------------------------------------------------------- #
def generate_pairs(
    speakers: Dict[str, List[str]],
    pairs_per_speaker: int,
    neg_pairs_per_combo: int,
) -> Tuple[List[Tuple[str, str, bool]], List[Tuple[str, str]]]:
    """
    Cria:
        • all_info  = [(f1, f2, expected_bool), ...]  (para relatório)
        • just_pairs = [(f1, f2), ...]                (para workers)
    Amostragem:
        • `pairs_per_speaker` pares positivos por locutor.
        • `neg_pairs_per_combo` pares negativos por combinação de locutores.
    """
    all_info: List[Tuple[str, str, bool]] = []

    # ---------- positivos (mesmo locutor) ----------
    for files in speakers.values():
        combos = list(itertools.combinations(files, 2))
        sampled = random.sample(combos, min(pairs_per_speaker, len(combos)))
        all_info.extend((f1, f2, True) for f1, f2 in sampled)

    # ---------- negativos (locutores diferentes) ----------
    for spk1, spk2 in itertools.combinations(speakers.keys(), 2):
        files1, files2 = speakers[spk1], speakers[spk2]
        all_neg_combos = [(f1, f2) for f1 in files1 for f2 in files2]
        sampled = random.sample(
            all_neg_combos, min(neg_pairs_per_combo, len(all_neg_combos))
        )
        all_info.extend((f1, f2, False) for f1, f2 in sampled)

    just_pairs = [(f1, f2) for f1, f2, _ in all_info]
    return all_info, just_pairs


# --------------------------------------------------------------------------- #
# 3. Divisão em lotes                                                         #
# --------------------------------------------------------------------------- #
def chunkify(seq: List[Tuple[str, str]], n_chunks: int) -> List[List[Tuple[str, str]]]:
    """Divide `seq` em `n_chunks` blocos de tamanhos o mais homogêneos possível."""
    k = len(seq)
    step = (k + n_chunks - 1) // n_chunks  # teto(k / n_chunks)
    return [seq[i : i + step] for i in range(0, k, step)]


# --------------------------------------------------------------------------- #
# 4. Worker paralelo                                                          #
# --------------------------------------------------------------------------- #
def _verify_batch(args: Tuple[str, List[Tuple[str, str]]]) -> List[bool]:
    """
    Executa verify_speakers(model, f1, f2) para um lote de pares.

    Cada processo:
      • carrega o modelo somente uma vez (cache global _MODEL)
      • devolve a lista de predições do seu lote
    """
    ckpt, batch = args
    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = load_model()  # type: ignore[attr-defined]
    model = _MODEL
    return [verify_speakers(model, f1, f2) for f1, f2 in batch]


# --------------------------------------------------------------------------- #
# 5. Avaliação e métricas                                                     #
# --------------------------------------------------------------------------- #
def evaluate(all_info: List[Tuple[str, str, bool]], preds: List[bool]) -> None:
    """Imprime accuracy, matriz de confusão e lista de erros."""
    y_true = [exp for _, _, exp in all_info]
    y_pred = preds

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
        description="Avalia accuracy de verificação de locutor "
        "paralelizando verify_speakers e permitindo balancear pares."
    )
    parser.add_argument(
        "--data_dir",
        default="audios",
        help="Diretório com subpastas por locutor contendo arquivos .wav.",
    )
    parser.add_argument(
        "--model_ckpt",
        default="ecapa_pretrained",
        help="Checkpoint ou identificador do modelo para load_model().",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Número de processos paralelos (padrão = CPUs disponíveis).",
    )
    parser.add_argument(
        "--pairs_per_speaker",
        type=int,
        default=10,
        help="Quantos pares positivos amostrar por locutor (padrão = 2).",
    )
    parser.add_argument(
        "--neg_pairs_per_combo",
        type=int,
        default=10,
        help="Quantos pares negativos amostrar por combinação de dois locutores "
        "(padrão = 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do gerador aleatório para reprodutibilidade.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ---------- coleta dos áudios ----------
    speakers = collect_audio_files(args.data_dir)
    if len(speakers) < 2:
        raise SystemExit("⚠️  É necessário ter pelo menos dois locutores com áudios.")

    total_wavs = sum(map(len, speakers.values()))
    print(
        f"Locutores: {len(speakers)}  |  Áudios: {total_wavs}  "
        f"|  Workers: {args.workers}"
    )

    # ---------- gera pares ----------
    all_info, just_pairs = generate_pairs(
        speakers,
        pairs_per_speaker=args.pairs_per_speaker,
        neg_pairs_per_combo=args.neg_pairs_per_combo,
    )
    print(f"Pares selecionados para avaliação: {len(just_pairs)}")

    # ---------- distribui em lotes ----------
    batches = chunkify(just_pairs, max(1, args.workers))
    print("⏳ Iniciando verificações em paralelo…")

    # ---------- verificação ----------
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = pool.map(
            _verify_batch, [(args.model_ckpt, batch) for batch in batches]
        )
    preds = [pred for batch_preds in futures for pred in batch_preds]
    print("✅ Verificações concluídas.")

    # ---------- relatório ----------
    evaluate(all_info, preds)


if __name__ == "__main__":
    main()