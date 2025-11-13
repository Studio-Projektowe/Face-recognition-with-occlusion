import csv
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from config import BASE_FOLDER_LOCAL # Import tylko dla spójności, nieużywany

def calculate_verification_metrics(csv_file):
    print(f"Wczytywanie wyników weryfikacji z: {csv_file}")
    
    scores = []
    labels = [] # 1 dla 'genuine', 0 dla 'imposter'
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scores.append(float(row['score']))
                    labels.append(1 if row['label'] == 'genuine' else 0)
                except (ValueError, TypeError):
                    print(f"Pominięto błędny wiersz: {row}")
                    
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {csv_file}")
        print("Upewnij się, że najpierw uruchomiłeś 'run_verification.py'.")
        sys.exit(1)
        
    if not labels:
        print("BŁĄD: Plik CSV jest pusty. Nie ma danych do analizy.")
        return

    y_true = np.array(labels)
    y_scores = np.array(scores)
    
    num_genuine = np.sum(y_true == 1)
    num_imposter = np.sum(y_true == 0)

    print("\n--- Wyniki Weryfikacji (1:1) ---")
    print(f"Całkowita liczba par: {len(y_true)}")
    print(f"Pary 'Genuine' (ja vs ja):    {num_genuine}")
    print(f"Pary 'Imposter' (ja vs obcy): {num_imposter}")
    
    if num_genuine == 0 or num_imposter == 0:
        print("BŁĄD: Do obliczenia ROC-AUC potrzeba zarówno par genuine, jak i imposter.")
        return

    # --- 1. Metryka: ROC-AUC ---
    # Ogólna jakość embeddingów
    try:
        auc = roc_auc_score(y_true, y_scores)
        print(f"\n--- Metryka: ROC-AUC ---")
        print(f"ROC-AUC: {auc:.6f}")
        print("(Im bliżej 1.0, tym model jest lepszy w odróżnianiu osób)")
    except ValueError as e:
        print(f"\nBŁĄD: Nie można obliczyć ROC-AUC: {e}")

    # --- 2. Metryka: Verification Accuracy @ Progi ---
    # Proste "działa/nie działa"
    print(f"\n--- Metryka: Verification Accuracy (Celność Weryfikacji) ---")
    thresholds_to_test = [0.5, 0.6, 0.7] # Progi zasugerowane przez Ciebie
    
    for threshold in thresholds_to_test:
        # Przewidujemy '1' (ta sama osoba) jeśli wynik jest POWYŻEJ progu
        y_pred = (y_scores >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        
        # Obliczamy dodatkowe statystyki dla tego progu
        tp = np.sum((y_pred == 1) & (y_true == 1)) # True Positive
        tn = np.sum((y_pred == 0) & (y_true == 0)) # True Negative
        fp = np.sum((y_pred == 1) & (y_true == 0)) # False Positive
        fn = np.sum((y_pred == 0) & (y_true == 1)) # False Negative
        
        print(f"  Celność @ Próg {threshold:.1f}: {acc * 100:.2f}%")
        print(f"    (TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn})")
        
    # --- 3. Metryka: TAR @ FAR ---
    # Naukowa ocena jakości
    print(f"\n--- Metryka: TAR @ FAR (True Accept Rate @ False Accept Rate) ---")
    
    # Obliczamy krzywą ROC
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_scores)
    # FPR = FAR (False Accept Rate)
    # TPR = TAR (True Accept Rate)
    
    far_targets = [0.1, 0.01, 0.001] # 10%, 1%, 0.1%
    
    for far_target in far_targets:
        try:
            # Znajdź pierwszy indeks, gdzie FPR jest <= nasz cel
            # (np. 0.01)
            # Używamy np.searchsorted, aby znaleźć, gdzie wstawić wartość
            # To jest szybsze niż pętla
            
            # Musimy odwrócić tablice, bo roc_curve sortuje malejąco
            if fpr_all[0] < fpr_all[-1]: # Jeśli posortowane rosnąco
                 idx = np.searchsorted(fpr_all, far_target, side='right') - 1
            else: # Jeśli posortowane malejąco (częściej)
                idx = np.searchsorted(fpr_all[::-1], far_target, side='right') - 1
                idx = len(fpr_all) - 1 - idx # Konwertuj z powrotem
            
            if idx < 0:
                 idx = 0 # Na wszelki wypadek

            tar_at_far = tpr_all[idx]
            threshold_for_far = thresholds_all[idx]
            
            print(f"  TAR @ FAR = {far_target * 100:g}% : {tar_at_far * 100:.2f}% (przy progu ~{threshold_for_far:.4f})")
            
        except Exception as e:
            print(f"Błąd przy obliczaniu TAR@FAR={far_target}: {e}")
            
    print("(Mówi nam, jak dobry jest system: np. 'Przy 1% fałszywych alarmów, system poprawnie rozpoznaje 95% prawdziwych użytkowników')")
    

if __name__ == "__main__":
    calculate_verification_metrics("verification_scores.csv")