import csv
import os
from collections import defaultdict
from dataclasses import dataclass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "hardest_ids_summary.csv")


@dataclass
class IdStats:
    total: int = 0
    false_count: int = 0

    @property
    def false_rate(self) -> float:
        return self.false_count / self.total if self.total else 0.0


def find_eval_csvs(base_dir: str):
    results = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name == "evaluation_results.csv":
                results.append(os.path.join(root, name))
    return results


def get_model_name(csv_path: str) -> str:
    parts = csv_path.split(os.sep)
    # models/<ModelName>/score/evaluation_results.csv
    try:
        idx = parts.index("models")
        return parts[idx + 1]
    except Exception:
        return os.path.basename(os.path.dirname(csv_path))


def infer_label_column(fieldnames):
    for candidate in ("found", "found_in_top3", "is_correct_top1"):
        if candidate in fieldnames:
            return candidate
    return None


def is_false(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"false", "0", "no", "n"}


def analyze_csv(csv_path: str):
    stats = defaultdict(IdStats)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return stats
        label_col = infer_label_column(reader.fieldnames)
        if not label_col or "query_id" not in reader.fieldnames:
            return stats

        for row in reader:
            qid = row.get("query_id", "").strip()
            if not qid:
                continue
            stats[qid].total += 1
            if is_false(row.get(label_col)):
                stats[qid].false_count += 1
    return stats


def top_n_hardest(stats: dict, n: int = 10):
    items = [
        (qid, s.false_count, s.total, s.false_rate)
        for qid, s in stats.items()
    ]
    items.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
    return items[:n]


def main():
    csv_paths = find_eval_csvs(BASE_DIR)
    if not csv_paths:
        print("Brak plików evaluation_results.csv w models/**/score")
        return

    summary_rows = []
    for path in csv_paths:
        model_name = get_model_name(path)
        stats = analyze_csv(path)
        top10 = top_n_hardest(stats, n=10)

        print(f"\nModel: {model_name}")
        print("ID\tFalse\tTotal\tFalseRate")
        for qid, false_count, total, rate in top10:
            print(f"{qid}\t{false_count}\t{total}\t{rate:.3f}")
            summary_rows.append({
                "model": model_name,
                "query_id": qid,
                "false_count": false_count,
                "total": total,
                "false_rate": f"{rate:.6f}",
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "query_id", "false_count", "total", "false_rate"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nZapisano podsumowanie do: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
