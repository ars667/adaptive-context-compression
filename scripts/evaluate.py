"""
Evaluation script: runs all 3 RAG modes on the QA dataset and compares metrics.

Usage:
    python scripts/evaluate.py [--questions N] [--output data/results/evaluation_results.json]
"""

import json
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline


MODES = ["baseline", "full_rag", "compressed"]
DATASET_PATH = "data/dataset/qa_dataset.json"
RESULTS_PATH = "data/results/evaluation_results.json"


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 3)


def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L score using the rouge-score library."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return round(scores["rougeL"].fmeasure, 3)
    except Exception:
        return None


def run_evaluation(n_questions: int | None = None, output_path: str = RESULTS_PATH):
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    if n_questions:
        dataset = dataset[:n_questions]

    print(f"Загрузка пайплайна...")
    pipeline = RAGPipeline()

    results = []
    summaries = {mode: {"tokens": [], "time_ms": [], "khr": [], "rouge_l": []} for mode in MODES}

    print(f"\nОцениваем {len(dataset)} вопросов в {len(MODES)} режимах...\n")

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        reference = item["reference_answer"]
        keywords = item.get("keywords", [])
        topic = item.get("topic", "—")

        print(f"[{qid:2d}/{len(dataset)}] {question[:60]}...")

        row = {"id": qid, "question": question, "topic": topic, "modes": {}}

        for mode in MODES:
            start = time.perf_counter()
            result = pipeline.query(question, mode=mode)
            elapsed_ms = (time.perf_counter() - start) * 1000

            answer = result["answer"]
            tokens = result["tokens_used"]
            compression_ratio = result.get("compression_ratio")

            khr = keyword_hit_rate(answer, keywords)
            rl = rouge_l(answer, reference)

            row["modes"][mode] = {
                "tokens_used": tokens,
                "time_ms": round(elapsed_ms, 1),
                "compression_ratio": compression_ratio,
                "keyword_hit_rate": khr,
                "rouge_l": rl,
                "answer": answer,
            }

            summaries[mode]["tokens"].append(tokens)
            summaries[mode]["time_ms"].append(elapsed_ms)
            summaries[mode]["khr"].append(khr)
            if rl is not None:
                summaries[mode]["rouge_l"].append(rl)

            cr_str = f" (CR={compression_ratio:.2f})" if compression_ratio else ""
            print(f"      {mode:12s}: {tokens:5d} tokens, {elapsed_ms:6.0f}ms, KHR={khr:.2f}{cr_str}")

        results.append(row)

    # Compute averages
    avg = {}
    for mode in MODES:
        s = summaries[mode]
        avg[mode] = {
            "avg_tokens": round(sum(s["tokens"]) / len(s["tokens"]), 1),
            "avg_time_ms": round(sum(s["time_ms"]) / len(s["time_ms"]), 1),
            "avg_khr": round(sum(s["khr"]) / len(s["khr"]), 3),
            "avg_rouge_l": round(sum(s["rouge_l"]) / len(s["rouge_l"]), 3) if s["rouge_l"] else None,
        }

    output = {"questions": results, "summary": avg}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 70)
    print(f"{'Режим':<14} {'Avg токены':>12} {'Avg время, мс':>14} {'Avg KHR':>9} {'Avg ROUGE-L':>12}")
    print("-" * 70)
    for mode in MODES:
        a = avg[mode]
        rl_str = f"{a['avg_rouge_l']:.3f}" if a["avg_rouge_l"] is not None else "  N/A"
        print(f"{mode:<14} {a['avg_tokens']:>12.1f} {a['avg_time_ms']:>14.1f} {a['avg_khr']:>9.3f} {rl_str:>12}")

    print("\nЭкономия токенов (compressed vs full_rag):")
    t_rag = avg["full_rag"]["avg_tokens"]
    t_comp = avg["compressed"]["avg_tokens"]
    if t_rag > 0:
        savings = (1 - t_comp / t_rag) * 100
        ratio = t_rag / t_comp if t_comp > 0 else float("inf")
        print(f"  full_rag: {t_rag:.0f} токенов → compressed: {t_comp:.0f} токенов")
        print(f"  Экономия: {savings:.1f}%  (в {ratio:.2f}x меньше токенов)")

    print(f"\nРезультаты сохранены: {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline modes")
    parser.add_argument("--questions", type=int, default=None, help="Сколько вопросов использовать (по умолчанию все)")
    parser.add_argument("--output", default=RESULTS_PATH, help="Путь для сохранения результатов")
    args = parser.parse_args()

    run_evaluation(n_questions=args.questions, output_path=args.output)
