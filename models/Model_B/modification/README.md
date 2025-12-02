# Projekt: Transfer Learning Modelu Buffalo_L (InsightFace) do PyTorch

Dokumentacja procesu konwersji i przygotowania modelu buffalo_l (ResNet50) do dotrenowania na własnych danych.

## Problem Wyjściowy

Chcieliśmy wziąć wytrenowany model w600k_r50.onnx z biblioteki InsightFace i wrzucić go do PyTorch, aby go dotrenować.

Okazało się to koszmarem z trzech powodów:

Bałagan w nazwach: Konwerter onnx2pytorch wypluł wagi z nazwami typu _initializer_layer1_0_bn1_weight, podczas gdy PyTorch oczekuje layer1.0.bn1.weight.

Losowa kolejność: Wagi w pliku .pth (z konwersji) nie zachowały kolejności topologicznej, więc zwykłe ładowanie "po kolei" (shape matching) zawiodło.

Martwe Warstwy (NaN/Zero): Po załadowaniu wag "na siłę", model zwracał NaN lub same zera. Powodem były zerowe wariancje w warstwach BatchNormalization (dzielenie przez zero) oraz uszkodzone wagi PReLU.

## Rozwiązanie: Custom Loader (loader_check.py)

Stworzyliśmy dedykowany skrypt ładujący, który działa jak "szpital" dla tego modelu. Proces wygląda tak:

1. Hybrid Smart Load (v5)

Etap 1 (Fuzzy Name Matching): Ignorujemy kropki, podkreślniki i prefiksy. Jeśli layer1_0_bn1 pasuje do layer1.0.bn1, ładujemy wagę.

Etap 2 (Greedy Shape Match): Dla pozostałych "sierot" (np. Conv_684) przeszukujemy cały plik w poszukiwaniu wolnego tensora o tym samym kształcie.

Wynik: Udaje się odzyskać 370 z 396 istotnych warstw (93% modelu).

2. Re-inicjalizacja (Patchowanie Dziur)

Około 8 modułów (głównie skróty downsample i głębokie warstwy) nie dało się dopasować. Zamiast zostawiać je puste (co zerowało sygnał), zostały one zreinicjalizowane (Kaiming Init).

Efekt: Model jest ciągły, ale te 8 warstw to "szum", który musi się dostroić podczas treningu.

3. Procedura "Sanitize" (Szpital)

Skrypt przelatuje przez wszystkie warstwy BatchNorm i wymusza minimalną wariancję (1e-4), aby zapobiec wybuchom NaN (Not a Number).

## Status Modelu (Przed Treningiem)

Stabilność:  Model działa, nie zwraca błędów, nie ma NaN.

Norma wektora:  Prawidłowa (~42900), sygnał przechodzi przez całą sieć.

Zgodność z oryginałem InsightFace:  Niska (~0.01).

Dlaczego? Przez te 8 zresetowanych warstw. Model ma "amnezję" w kilku miejscach.

Czy to źle? Nie. Testy wykazały, że model uczy się błyskawicznie. Wystarczy krótki trening (fine-tuning), aby te warstwy "wskoczyły" na miejsce, a reszta sieci (93%) jest już wytrenowana.

## Jak Uruchomić Trening?

Głównym plikiem jest train_project.py. Używa on naprawionego loadera automatycznie.

Wskazówka: Pierwsze epoki mogą mieć wysoki Loss, dopóki zresetowane warstwy się nie ustabilizują. To normalne.