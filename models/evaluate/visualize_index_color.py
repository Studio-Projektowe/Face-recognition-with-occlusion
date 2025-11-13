# pip install scikit-learn matplotlib

import faiss
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Importujemy do obsługi map kolorów

def visualize_faiss_index(index_file, map_file, output_image):
    print(f"Wczytywanie indeksu z {index_file}...")
    try:
        index = faiss.read_index(index_file)
    except Exception as e:
        print(f"BŁĄD: Nie można wczytać pliku {index_file}. Uruchom najpierw budowanie galerii.")
        print(f"Error: {e}")
        return

    print(f"Wczytywanie mapowania ID z {map_file}...")
    try:
        with open(map_file, 'r') as f:
            id_map = json.load(f)
    except Exception as e:
        print(f"BŁĄD: Nie można wczytać pliku {map_file}.")
        print(f"Error: {e}")
        return

    # Krok 1: Wyciągnij wektory z indeksu FAISS
    num_vectors = index.ntotal
    dimension = index.d
    
    if num_vectors == 0:
        print("BŁĄD: Indeks jest pusty (nie zawiera wektorów).")
        return

    print(f"Wyciąganie {num_vectors} wektorów (o wymiarze {dimension}) z indeksu...")
    try:
        vectors = index.reconstruct_n(0, num_vectors)
    except RuntimeError:
        print("BŁĄD: Tego typu indeks FAISS nie wspiera 'reconstruct_n'.")
        print("Upewnij się, że używasz IndexFlatIP.")
        return

    # Krok 2: Przygotuj etykiety i kolory
    # Tworzymy listę etykiet ['id_3', 'id_5', ...] we właściwej kolejności
    labels = [id_map.get(str(i), f"ID_{i}?") for i in range(num_vectors)]

    # Generujemy unikalne kolory dla każdego ID
    # Możemy użyć mapy kolorów z matplotlib
    colors = cm.get_cmap('tab20', len(labels)) # 'tab20' to dobra mapa dla wielu kategorii
    point_colors = [colors(i) for i in range(len(labels))]

    # Krok 3: Redukcja wymiarowości (t-SNE)
    print("Uruchamiam t-SNE, aby zredukować wymiary z 512 do 2...")
    print("To może potrwać chwilę...")
    
    perplexity_value = min(30.0, float(num_vectors - 1))
    if perplexity_value <= 1:
        print("Zbyt mało danych do t-SNE (potrzeba co najmniej 2 wektorów).")
        return

    tsne = TSNE(n_components=2, 
                perplexity=perplexity_value, 
                max_iter=1000, 
                random_state=42, 
                init='pca', 
                learning_rate='auto')
    
    vectors_2d = tsne.fit_transform(vectors)

    # Krok 4: Rysowanie wykresu
    print(f"Tworzenie wykresu i zapisywanie do {output_image}...")
    plt.figure(figsize=(16, 12))
    
    # Rozrzucamy punkty na wykresie, przypisując każdemu unikalny kolor
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=point_colors, alpha=0.7)
    
    # Zamiast etykiet tekstowych przy punktach, dodajemy legendę
    # Tworzymy fałszywe uchwyty do legendy, aby wyświetlić nazwy ID i odpowiadające im kolory
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10)
                       for label, color in zip(labels, point_colors)]
    
    # Umieszczamy legendę poza głównym obszarem wykresu, aby nie zasłaniała punktów
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="ID Wektorów")


    plt.title("Wizualizacja t-SNE 'średnich' wektorów tożsamości z galerii FAISS (kolorowane ID)")
    plt.xlabel("Wymiar t-SNE 1")
    plt.ylabel("Wymiar t-SNE 2")
    plt.grid(True)
    
    # Zapisz plik
    plt.savefig(output_image, bbox_inches='tight', dpi=150)
    print(f"Gotowe! Wizualizacja zapisana w {output_image}")

if __name__ == "__main__":
    visualize_faiss_index(
        index_file="gallery.index", 
        map_file="gallery_id_map.json", 
        output_image="gallery_visualization_colored.png" # Zmieniona nazwa pliku wyjściowego
    )