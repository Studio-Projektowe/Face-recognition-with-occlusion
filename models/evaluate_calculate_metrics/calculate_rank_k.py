import csv
import sys
from models.ArcFace_Large.evaluation.config import RESULTS_CSV # Importuje nazwę pliku z config

def calculate_rank_k_accuracy(csv_file):
    print(f"Wczytywanie wyników z: {csv_file}")
    
    total_queries = 0
    rank1_correct = 0
    rank3_correct = 0
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Oczyszczenie (usunięcie spacji, jeśli są)
                query_id = row['query_id'].strip()
                top1_id = row['top1_id'].strip()
                top2_id = row['top2_id'].strip()
                top3_id = row['top3_id'].strip()
                
                total_queries += 1
                
                # Sprawdzenie Rank-1
                if query_id == top1_id:
                    rank1_correct += 1
                
                # Sprawdzenie Rank-3 (czy poprawny ID jest w Top 3)
                if query_id in (top1_id, top2_id, top3_id):
                    rank3_correct += 1
                    
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {csv_file}")
        print("Upewnij się, że najpierw uruchomiłeś 'run_evaluation.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"BŁĄD: Wystąpił błąd podczas czytania pliku CSV: {e}")
        sys.exit(1)
        
    if total_queries == 0:
        print("BŁĄD: Plik CSV jest pusty. Nie ma danych do analizy.")
        return

    # Obliczenie i wydrukowanie wyników
    rank1_acc = (rank1_correct / total_queries) * 100
    rank3_acc = (rank3_correct / total_queries) * 100
    
    print("\n--- Wyniki Identyfikacji (1:N) ---")
    print(f"Całkowita liczba zapytań: {total_queries}")
    print(f"Poprawne trafienia Rank-1:  {rank1_correct}")
    print(f"Poprawne trafienia Rank-3:  {rank3_correct}")
    print("---")
    print(f"Rank-1 Accuracy (Recall@1): {rank1_acc:.2f}%")
    print(f"Rank-3 Accuracy (Recall@3): {rank3_acc:.2f}%")

if __name__ == "__main__":
    calculate_rank_k_accuracy(RESULTS_CSV)