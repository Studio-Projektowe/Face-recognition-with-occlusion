import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve # Do obliczenia EER

# --- Konfiguracja ---

# Plik wejściowy wygenerowany przez Twój poprzedni skrypt
INPUT_CSV = "verification_scores.csv"

# Docelowe poziomy FAR (False Accept Rate), które chcemy analizować
# 0.1 = 10%, 0.01 = 1%, 0.001 = 0.1%
TARGET_FARS = [0.1, 0.01, 0.001]

# --- Koniec Konfiguracji ---


def calculate_tar_at_far(genuine_scores, imposter_scores, target_fars):
    """
    Oblicza True Accept Rate (TAR) dla określonych poziomów False Accept Rate (FAR).
    """
    print("--- 1. Obliczanie metryk TAR @ FAR ---")
    
    total_imposters = len(imposter_scores)
    total_genuines = len(genuine_scores)
    
    if total_imposters == 0 or total_genuines == 0:
        print("BŁĄD: Nie można obliczyć metryk. Brak wyników 'genuine' lub 'imposter'.")
        return

    print(f"Znaleziono {total_genuines} par 'genuine'.")
    print(f"Znaleziono {total_imposters} par 'imposter'.\n")

    for far_target in target_fars:
        # Jaki próg (threshold) daje nam ten docelowy FAR?
        # Musimy znaleźć odpowiedni percentyl.
        # Dla FAR = 1% (0.01), chcemy próg, który jest wyższy niż 99% (1.0 - 0.01) wyników imposter.
        percentile = (1.0 - far_target) * 100
        
        # Znajdź ten próg w zbiorze wyników imposter
        threshold = np.percentile(imposter_scores, percentile)
        
        # Teraz, gdy mamy próg, sprawdźmy, ilu PRAWDZIWYCH użytkowników 
        # zostało poprawnie zaakceptowanych (ich wynik był >= próg).
        true_accepts = np.sum(genuine_scores >= threshold)
        
        # Oblicz TAR (True Accept Rate)
        tar = true_accepts / total_genuines
        
        # Prezentacja wyników
        print(f"📊 TAR @ FAR = {far_target*100: <5}% : {tar*100:,.2f}% (przy progu ~{threshold:.4f})")
        
    print("\nInterpretacja:")
    print("Powyższe linie czytaj jako: 'Aby osiągnąć poziom bezpieczeństwa FAR (np. 1% pomyłek oszustów),")
    print("nasz system poprawnie rozpoznaje X% prawdziwych użytkowników (TAR)'.")

def calculate_eer(genuine_scores, imposter_scores):
    """
    Oblicza Equal Error Rate (EER) - punkt, w którym FAR == FRR.
    To jest odpowiedź na pytanie "jak bardzo nakładają się na siebie wyniki genuine i imposter".
    """
    print("\n--- 2. Obliczanie Equal Error Rate (EER) ---")

    # Tworzymy tablice dla funkcji roc_curve
    # 1 = genuine (positive), 0 = imposter (negative)
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    # Obliczamy krzywą ROC
    # fpr = False Positive Rate (to jest to samo co FAR)
    # tpr = True Positive Rate (to jest to samo co TAR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Obliczamy FRR (False Reject Rate)
    # FRR = 1 - TAR (czyli 1 - tpr)
    fnr = 1 - tpr # FNR to to samo co FRR
    
    # Szukamy punktu, gdzie różnica między FAR i FRR jest najmniejsza
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    
    # Pobieramy wartości w tym punkcie
    # Bierzemy średnią z FAR i FRR w tym punkto, aby dostać EER
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    
    print(f"📈 Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"   (Osiągnięty przy progu: ~{eer_threshold:.4f})")
    print("\nInterpretacja:")
    print("EER to 'najlepszy' możliwy próg dla systemu, gdzie myli się on ")
    print("tak samo często odrzucając prawdziwych użytkowników, jak akceptując oszustów.")
    print("Im niższy EER, tym lepszy system (mniejsze nakładanie się wyników).")


def main():
    print(f"Wczytywanie pliku {INPUT_CSV}...")
    
    if not os.path.exists(INPUT_CSV):
        print(f"BŁĄD: Nie znaleziono pliku {INPUT_CSV}.")
        print("Upewnij się, że ten skrypt znajduje się w tym samym folderze co plik z wynikami.")
        return

    try:
        # Wczytywanie danych jest najszybsze w ten sposób
        df = pd.read_csv(INPUT_CSV)
        
        # Optymalizacja: Przekształć w tablice NumPy do szybkich obliczeń
        # To jest kluczowy krok dla wydajności.
        print("Separowanie wyników 'genuine' i 'imposter'...")
        genuine_scores = df[df['label'] == 'genuine']['score'].to_numpy()
        imposter_scores = df[df['label'] == 'imposter']['score'].to_numpy()
        
        # Usuń DataFrame, aby zwolnić pamięć
        del df 
        
        calculate_tar_at_far(genuine_scores, imposter_scores, TARGET_FARS)
        calculate_eer(genuine_scores, imposter_scores)

    except ImportError:
        print("\nBŁĄD KRYTYCZNY: Nie znaleziono biblioteki 'scikit-learn'.")
        print("Proszę ją zainstalować: pip install scikit-learn")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd: {e}")
        print("Upewnij się, że plik CSV ma kolumny 'score' i 'label'.")

if __name__ == "__main__":
    main()