# Projekt: Identyfikacja tożsamości po zdjęciu twarzy z okluzją

## Sformułowanie Problemu Badawczego (Cel Projektu)

### Definicja Problemu

Głównym problemem badawczym jest identyfikacja tożsamości osób na podstawie zdjęć twarzy z częściową okluzją, ze szczególnym uwzględnieniem obszaru oczu (np. przez okulary, zasłonięcia w monitoringu, czy dynamicznie nakładany pasek). Chociaż standardowe algorytmy rozpoznawania twarzy (Face Recognition) osiągają wysoką skuteczność w warunkach idealnych, ich działanie drastycznie spada, gdy kluczowe cechy biometryczne, takie jak oczy, są częściowo zasłonięte.

W kontekście systemów bezpieczeństwa, weryfikacji tożsamości (np. w telefonach) oraz monitoringu wizyjnego, jest to istotna luka, którą projekt ma za zadanie zminimalizować

### Cel Główny Projektu (Rozwiązanie)

Celem jest zaprojektowanie, implementacja i porównanie efektywności dwóch metod uczenia maszynowego (Model A: Baseline vs. Model B: Modyfikacja autorska) do zadania identyfikacji osób, których twarz jest częściowo zasłonięta. Badamy, czy wprowadzenie modyfikacji architektoniczno-treningowych (Moduł Uwagi i Auxiliary Loss) poprawi odporność modelu ArcFace na okluzję w porównaniu do silnego baseline'u.

### Hipoteza badawcza

Dodanie modułu uwagi oraz auxiliary loss do modelu ArcFace (Buffalo_L) zwiększy odporność systemu rozpoznawania twarzy na okluzję oczu i poprawi skuteczność identyfikacji 1:N w porównaniu z modelem bazowym fine-tuningowanym wyłącznie na danych z okluzją.

### Kluczowe Założenia Metodyczne

Założenia te są fundamentalne dla Waszej metodyki ewaluacji (Identyfikacja 1:N)

#### Zadanie: 

Projekt koncentruje się na identyfikacji tożsamości (czyli na pytaniu: "Kim jest ta osoba z okluzją?"), a nie tylko na weryfikacji (czyli na pytaniu: "Czy ta osoba jest osobą X?").

#### Założenie o Galerii (Wiedza Bazowa): 

Przyjmuje się, że system ma dostęp do co najmniej jednego zdjęcia referencyjnego (galerii) każdej osoby, której tożsamość ma być rozpoznana, przy czym te zdjęcia referencyjne są wolne od okluzji.

#### Proces Ewaluacji (Testowanie):

Model generuje wektory cech (embeddingi) na podstawie zdjęć referencyjnych bez okluzji i zapisuje je w galerii FAISS. Następnie, model generuje embeddingi dla zdjęć-zapytań z okluzją oczu i przeszukuje galerię FAISS, aby znaleźć 3 najbliższych sąsiadów i ustalić tożsamość.

To podejście w pełni symuluje scenariusz, w którym np. użytkownik konfiguruje Face ID bez okularów, a system ma go rozpoznać, gdy później okulary założy.

### Ważne wymagania

* Wejście modelu: zdjęcia twarzy w rozdzielczości 112×112.

* Wyjście modelu: wektor embeddingowy o długości 512.

* Dane treningowe: zbiór danych uporządkowany według ID osoby (klas).

* Każda tożsamość musi posiadać kilkanaście lub więcej zdjęć, najlepiej w różnych warunkach (pozycja, światło, mimika).

* Zbiór nie może być zbyt mały — inaczej model nie nauczy się prawidłowo rozpoznawać twarzy.

* Model bazowy: musi dać się załadować jako pełna architektura, tak aby można było wstawić dodatkową warstwę (np. moduł uwagi) i dalej trenować.

* Model musi być załadowany w formie umożliwiającej modyfikację architektury (np. w PyTorch). Nie trenujemy od zera — wykorzystujemy Buffalo_S, Buffalo_L lub VGGFace jako backbone.

* Preprocessing i wyrównanie twarzy (face alignment) muszą być identyczne jak w modelu bazowym — 5-punktowe landmarki + wycięcie do 112×112. Zmiana alignowania znacząco psuje jakość embeddingów.

* Normalizacja wejścia (mean, std lub skala do [-1, 1]) musi być zgodna z oryginalnym trenowaniem ArcFace; inaczej embeddingi nie będą porównywalne.

* Dodawany moduł uwagi musi działać wyłącznie wewnątrz sieci i nie może zmieniać rozmiaru końcowego embeddingu (512). W przeciwnym razie nie da się porównywać wyników i używać FAISS.

* Trening musi odbywać się z bardzo niskim learning rate, aby nie uszkodzić wcześniej wyuczonych wag ArcFace (np. 1e-4 dla backbone i 1e-3 dla nowych warstw).

* Okluzje muszą być nakładane losowo (różne pozycje, rozmiar, kształt, kolor), aby model uczył się odporności na różne typy zasłonięcia oczu.

* Galeria (zdjęcia referencyjne bez okluzji) musi być odseparowana od zbioru testowego z okluzją. Żadne zdjęcie nie może pojawić się w obu zbiorach, aby uniknąć przecieków.

* Metryka w FAISS musi być spójna (L2 lub cosine). Raz wybrana nie może być zmieniana w trakcie projektu.

* Trenujemy tylko backbone i nowy moduł — nie trenujemy ArcFace Softmax, ponieważ nie mamy oryginalnych wag klasyfikatora z setkami tysięcy klas.

## Ustalenia Wstępne

Wybrany model bazowy: ArcFace Buffalo_L (IResNet-50).

Framework: PyTorch – ze względu na większą elastyczność przy modyfikowaniu architektury (np. dodawanie modułu uwagi).

Model A (Baseline): ArcFace Buffalo_L fine-tuned na naszym zbiorze danych, z losową okluzją nakładaną na oczy podczas treningu.

Model B (Modyfikacja): Model A rozszerzony o Moduł Uwagi (Attention Module) oraz dodatkową stratę pomocniczą (Auxiliary Loss), mające zwiększyć odporność na okluzję.

## Kroki (od końca)

### 1. Napisanie raportu w LaTeX’ie

Raport końcowy będzie zawierał:

* kompletną dokumentację implementacji (architektura, kod, konfiguracje treningu),

* przegląd literatury i obecnego stanu wiedzy na temat ArcFace, modeli ResNet/iResNet, mechanizmów uwagi oraz odporności na okluzję,

* dokładnie określony cel i zakres projektu,

* opis metodologii, zastosowanych modyfikacji, sposobu augmentacji (okluzji) i struktury zbioru danych,

* szczegółową analizę wyników oraz porównanie testowanych metod,

* omówienie ograniczeń projektu i interpretację rezultatów,

* końcowe wnioski.

W raporcie musi się znaleźć kilka alternatywnych podejść do rozwiązania problemu, wraz z jasnym wskazaniem, które z nich okazało się najlepsze oraz dlaczego.

### 2. Ewaluacja końcowego modelu

#### Architektura modelu

* Model został zaimplementowany w PyTorch – framework ten umożliwia ingerencję w backbone (IResNet50 / Buffalo_L).

* Jako model bazowy użyto Buffalo_L (IResNet50) z InsightFace.

* Dodano mechanizm uwagi w końcowych częściach sieci (Stage 3 i Stage 4), ponieważ tam model przetwarza najbardziej abstrakcyjne cechy twarzy.

* Dodano dodatkową funkcję straty Auxiliary Loss, której celem jest stabilizacja i dodatkowe wymuszenie separowalności embeddingów.

* Ostatnia warstwa modelu to w pełni połączona warstwa (FC), która generuje embedding o wymiarze 512, zgodny z ArcFace.

#### Proces treningu

* Trening przeprowadzono na zbiorze danych uporządkowanym według tożsamości (train/val/test).

* Wszystkie obrazy zostały wyrównane (aligned) do 112×112 zgodnie z procedurą używaną przez ArcFace (5-punktowe landmarki).

* Normalizacja obrazów była identyczna jak w oryginalnym modelu Buffalo_L (np. skala [-1, 1]).

* W trakcie treningu aplikowano losowe okluzje oczu (różne pozycje, szerokości, kolory i intensywność).

* Trenowano wyłącznie backbone + moduł uwagi, nie trenowano softmaxa ArcFace (niewykorzystany w naszym zadaniu 1:N).

* Learning rate był niski, np. LR_backbone = 1e-4, LR_attention = 1e-3.

Ewaluacja metodą 1:N z użyciem FAISS

* Model generuje embeddingi 512 dla wszystkich zdjęć bez okluzji i zapisuje je jako galerię (baza referencyjna).

* embeddingi są przechowywane i indeksowane za pomocą FAISS (L2 lub cosine similarity – jedna metryka konsekwentnie).

* Dla zdjęć z okluzją:

    * model oblicza embedding,

    * FAISS zwraca 3 najbliższych sąsiadów z galerii,

    * podobieństwa są interpretowane jako dopasowanie do tożsamości.

Wyniki otrzymujemy w formie np.:

Obraz A → najbardziej podobny do osoby A (0.70), dalej B (0.10), C (0.05).

#### Wizualizacja

Embeddingi zostaną przedstawione w 2D za pomocą t-SNE lub UMAP, aby pokazać separowalność klas oraz wpływ okluzji.

#### Ewaluacja ilościowa (metryki)

Do oceny jakości rozpoznawania zastosujemy m.in.:

* Top-1 Accuracy / Top-3 Accuracy – skuteczność identyfikacji 1:N.

* TAR @ FAR=0.1% – True Accept Rate przy niskim False Accept Rate.

* EER (Equal Error Rate) – punkt, w którym błędy FA i FR są równe.

* ROC-AUC – pole pod krzywą ROC dla par pozytywne/negatywne.

* CMC curve (Cumulative Match Curve) – standard dla identyfikacji twarzy.

### 3. Modyfikacja modelu (Model B)

Mechanizm uwagi zostanie dodany do Stage 3 i Stage 4 IResNet-50.
W tych warstwach sieć uczy się najbardziej semantycznych cech twarzy, dlatego uwaga jest tu najbardziej efektywna.

Auxiliary Loss zostanie użyta jako dodatkowa funkcja straty, wspierającą właściwą separację embeddingów i stabilizację treningu.

### 4. Trening modelu bazowego (Model A – czysty Buffalo_L)

Zanim zostanie wprowadzona jakakolwiek modyfikacja, należy wytrenować czysty model Buffalo_L (IResNet50) na naszym zbiorze danych, aby:

* uzyskać baseline, do którego porównamy Model B,

* sprawdzić, jak sam backbone radzi sobie z okluzją oczu po fine-tuningu,

* ocenić wpływ dodawanego modułu uwagi.

#### Szczegóły treningu Modelu A

* Korzystamy z oryginalnej architektury Buffalo_L z InsightFace (PyTorch).

* Nie zmieniamy żadnych warstw.

* Trenujemy tylko backbone (embedding 512), nie trenujemy softmaxa ArcFace.

* Preprocessing, normalizacja i alignment są identyczne jak w oryginale.

* Używamy tej samej augmentacji okluzji, co później w Modelu B.

* Learning rate niski: ok. 1e-4.

* Strata: ArcFace Loss (w wersji do fine-tuningu embeddingu), ewentualnie prosty Triplet/Contrastive Loss.

### 5. Ewaluacja modelu bazowego (Model A)

Model A musi zostać oceniony przed modyfikacjami, aby stanowił punkt odniesienia.

Sposób ewaluacji (taki sam jak w Modelu B)

* Generujemy embeddingi 512 dla zdjęć bez okluzji → budujemy galerię FAISS.

* Dla obrazów z okluzją generujemy embeddingi i szukamy k-najbliższych sąsiadów (k=3).

* Otrzymujemy ranking tożsamości na podstawie podobieństwa kosinusowego lub L2.

Metryki dla Modelu A

* Top-1 / Top-3 Accuracy.

* TAR @ FAR=0.1%.

* EER.

* ROC-AUC.

* CMC curve.

#### Wizualizacja embeddingów (t-SNE/UMAP).

Cel ewaluacji Modelu A

* Ustalić wyjściową jakość systemu bez mechanizmu uwagi.

* Sprawdzić, jak okluzja wpływa na czysty Buffalo_L.

* Stworzyć punkt odniesienia, który Model B powinien poprawić.

### 6. Ewaluacja modeli pre-trained

Cele:

* Sprawdzić działanie dostępnych modeli Buffalo_L, Buffalo_S i VGGFace przed fine-tuningiem.

* Porównać ich baseline’ową skuteczność na naszym zbiorze testowym z okluzją i bez okluzji.

* Upewnić się, że modele generują embeddingi 512-d (Buffalo_L i Buffalo_S) lub zgodne z VGGFace (często 4096-d, można dostosować FC do 512-d dla spójności).

Proces:

* Załaduj pre-trained model w PyTorch (architektura + wagi).

* Zastosuj identyczny preprocessing i face alignment jak w treningu.

* Przetestuj model na zbiorze testowym, generując embeddingi.

* Dokonaj wstępnej ewaluacji z użyciem FAISS (Top-1, Top-3, TAR@FAR, EER, CMC).

* Zapisz wyniki dla porównania z późniejszym fine-tuningiem.

Ważne punkty:

* Upewnij się, że wagi pochodzą z PyTorch. InsightFace w repozytorium często daje PaddlePaddle / ONNX, więc mogą wymagać konwersji do PyTorch.

* Wymiar embeddingu musi być identyczny z tym, który będzie użyty w późniejszym fine-tuningu (512-d dla Buffalo_L/S).

* Sprawdź, czy model obsługuje te same kanały wejściowe i normalizację obrazów (mean/std lub [-1,1]).

### 7. Załadowanie modeli (architektura + wagi)

Cele:

* Przygotować model bazowy identyczny z tym, który będzie trenowany (fine-tuning Buffalo_L i modyfikacje Model B).

* Upewnić się, że zarówno architektura, jak i wagi są poprawnie załadowane.

Proces:

* Wybierz framework: PyTorch (ze względu na łatwość wstawiania mechanizmów uwagi).

* Załaduj architekturę modelu (np. Buffalo_L → IResNet50).

* Załaduj pre-trained wagi. Upewnij się, że:

* wagi odpowiadają wersji architektury,

* layer names pasują do modelu w PyTorch (czasem trzeba konwersji z Paddle/ONNX),

* embedding FC layer ma właściwy wymiar (512-d).

* Zablokuj lub “freeze” warstwy, których nie chcemy trenować od początku (np. wstępny backbone).

* Sprawdź poprawność działania: podaj kilka testowych zdjęć, upewnij się, że embeddingi są w oczekiwanym zakresie.

Ważne punkty:

* Normalizacja i alignment muszą być identyczne jak podczas pre-treningu.

* Jeśli w przyszłości dodajesz moduł uwagi, upewnij się, że nie zmienia wymiaru embeddingu.

* Każdy model (Buffalo_L, Buffalo_S, VGGFace) powinien być traktowany spójnie: te same preprocessing, face alignment i metryki.

* Zapisz stan modeli (architektura + wagi) po sprawdzeniu – będzie to punkt odniesienia dla późniejszego fine-tuningu i eksperymentów.

### 8. Przygotowanie funkcji do nakładania okluzji

Cel:

Tworzymy funkcję, która nakłada losowe okluzje na oczy (lub inne wybrane części twarzy) na obrazy, korzystając z zapisanych landmarków twarzy w pliku JSON.

Wymagania techniczne:

Funkcja powinna przyjmować:

* obraz twarzy,

* współrzędne landmarków (oczy, nos, usta itp.),

* parametry okluzji (szerokość, wysokość, kolor, przepuszczalność).

* Okluzja powinna być losowa w obrębie zadanego regionu:

* różna pozycja w obrębie oczu,

* różna szerokość i wysokość,

* różna przepuszczalność / intensywność / kolor.

* Zachowaj oryginalny rozmiar obrazu 112×112 i poprawnie wyrównaj twarz.

* Można użyć bibliotek: OpenCV, Pillow, Albumentations.

Funkcja powinna zwracać zarówno obraz zmodyfikowany, jak i oryginalny (dla porównań).

Użycie:

Funkcja będzie stosowana zarówno w czasie treningu (augmentacja) jak i przy ewaluacji (symulacja okluzji w zbiorze testowym).

### 9. Przygotowanie skryptów ewaluacyjnych

Cel:

Zanim załadujemy i fine-tunujemy modele, musimy mieć skrypty ewaluacyjne, aby sprawdzić:

* poprawność załadowanych modeli,

* baseline’ową skuteczność pre-trained modeli,

* spójność embeddingów i metryk.

Wymagania techniczne:

* Skrypt powinien przyjmować model w PyTorch i zbiór testowy (obraz + ID osoby).

* Musi generować embeddingi 512-d (dla Buffalo_L/S) lub odpowiednio dopasowane dla VGGFace.

Ewaluacja z użyciem FAISS:

* zbudowanie galerii embeddingów dla zdjęć bez okluzji,

* porównanie z embeddingami zdjęć z okluzją,

* wyznaczenie Top-1, Top-3 Accuracy, TAR@FAR, EER, CMC.

Wizualizacja embeddingów w 2D (t-SNE lub UMAP).

* Obsługa różnych metryk odległości: cosine similarity lub L2, konsekwentnie w całym pipeline.

* Skrypt powinien działać dla wszystkich pre-trained modeli: Buffalo_L, Buffalo_S, VGGFace.

Wynik:

Zestaw baseline’owych metryk i wizualizacji dla modeli pre-trained, które posłużą jako punkt odniesienia przed fine-tuningiem.

### 10. Podział zbioru danych i przygotowanie struktur katalogów

Cel:

Przygotować dane do treningu, walidacji i testów w sposób spójny z wymaganiami projektu, tak aby każda tożsamość była reprezentowana w odpowiednim podziale i każdy obraz miał przypisane landmarki.

#### Proces:

1. Podział według tożsamości:

Cały zbiór danych dzielimy w stosunku 80% train / 10% validation / 10% test.

Ważne: wszystkie zdjęcia jednej osoby trafiają do jednego podzbioru — brak przecieków między train/val/test.

Struktura folderów:

```
webface_112x112/
  train/
    id_001/
        001/
            001.jpg
            001.json
        002/
            002.jpg
            002.json
    id_002/
      ...
  val/
    id_010/
      ...
  test/
    id_020/
      ...
```

2. Parowanie zdjęć z landmarkami:

Każde zdjęcie twarzy ma odpowiadający plik JSON z 5-punktowymi landmarkami (oczy, nos, usta).

Pliki JSON muszą być identycznie nazwane jak obrazy (img_001.jpg ↔ img_001.json).

3. Sprawdzenie poprawności:

Wszystkie pliki obrazu mają odpowiadające landmarki.

Brak brakujących lub nadmiarowych plików.

Rozmiar obrazów dopasowany do wymagań modelu (112×112).

#### Ważne punkty:

Podział wg tożsamości zapewnia, że model nie “widzi” tej samej osoby w train i test.

Zachowanie struktury folderów ułatwia automatyczne ładowanie danych w PyTorch Dataset i aplikowanie augmentacji (np. losowych okluzji).

Możliwość łatwego generowania batchy train/val/test z przypisanymi landmarkami.

### 11. Pobranie i wstępne przetworzenie datasetu

Cel:

Przygotować dane wejściowe do projektu, pobierając wybrany dataset CASIA-WebFace w rozdzielczości 112×112 i przetwarzając je w sposób spójny z wymaganiami modelu (detekcja twarzy + zapis landmarków).

#### Proces:

1. Pobranie danych:

Dataset CASIA-WebFace dostępny na Kaggle lub w repozytoriach akademickich.

Wybrać wersję już przyciętą do 112×112, jeśli dostępna; jeśli nie, przyciąć obrazy ręcznie lub skryptem.

2. Detekcja i wyrównanie twarzy:

Użyć RetinaFace (lub innego sprawdzonego detektora z InsightFace).

Dla każdego obrazu wykryć twarz i 5-punktowe landmarki (oczy, nos, usta).

Wyrównać twarz do standardowego rozmiaru 112×112.

3. Zapis wyników:

Każde przetworzone zdjęcie zapisać w folderze datasetu.

Dla każdego obrazu wygenerować odpowiadający plik .json z landmarkami.

img_001.jpg
img_001.json


4. Sprawdzenie poprawności:

Każdy obraz ma odpowiadający plik JSON.

Brak brakujących danych lub błędów w detekcji twarzy.

Wszystkie obrazy są w rozmiarze 112×112 i gotowe do podziału na train/val/test.

#### Ważne punkty:

* Detekcja twarzy i wyrównanie muszą być identyczne dla wszystkich danych, aby embeddingi z modelu były spójne.

* Ten krok przygotowuje dane do kolejnego kroku 10 (podział na train/val/test i parowanie z landmarkami).

* Przygotowanie danych w tym kroku pozwala na łatwe stosowanie augmentacji, np. losowej okluzji podczas treningu.

### 12. Przygotowanie środowiska w GCP z Vertex AI

Cel:

Stworzyć stabilne środowisko do trenowania i ewaluacji modeli z GPU oraz zapewnić łatwy dostęp do datasetu i wyników.

#### Proces:

* Utworzenie projektu w Google Cloud Platform (GCP).

* Vertex AI Workbench / Notebooks:

* Utworzyć instancję VM w Vertex AI z GPU (np. NVIDIA Tesla T4, A100, V100).

* GPU umożliwia wykorzystanie PyTorch z akceleracją CUDA do trenowania modeli głębokiego uczenia.

Dysk współdzielony (Persistent Disk / Filestore):

* Podłączyć dysk do VM, na którym będzie przechowywany dataset oraz wyniki treningu.

* Dysk musi mieć wystarczającą pojemność na cały dataset (CASIA-WebFace ~ 1–2 GB przy 112×112), wagi modeli, pliki JSON z landmarkami oraz wyniki eksperymentów.

Konfiguracja środowiska:

* Zainstalować Python 3.9+, PyTorch z obsługą CUDA, FAISS, OpenCV, Albumentations i inne potrzebne biblioteki (InsightFace, tqdm, pandas, scikit-learn itp.).

Sprawdzić, czy GPU jest dostępne w PyTorch:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Upewnić się, że wszystkie pakiety i sterowniki GPU są kompatybilne.

Ważne punkty:

* Wszystkie eksperymenty będą wykonywane na tej samej VM, aby zapewnić spójność wyników.

* Dysk współdzielony umożliwia łatwy dostęp do datasetu i zapis wyników między restartami VM.

* GPU pozwala na znacznie szybsze trenowanie modeli deep learningowych w PyTorch.

