Sekcja Modelu | Warstwa / Moduł | Skąd są wagi? | Status Treningu |Komentarz |
--------------|-----------------|---------------|-----------------|----------|
STEM | conv1 (Wejście) | Załadowane| Zamrożone|Widzi podstawowe krawędzie/kolory. 
STEM | bn1, prelu | Załadowane| Trenowane|BN i PReLU odmrażasz w całym modelu. 
LAYER 1 | Bloki 0, 1 | Załadowane| Zamrożone|Conv zamrożone, BN/PReLU odmrożone. 
LAYER 1 |Blok 2 (Ostatni) | Częściowo Losowe| Trenowane|bn3 jest losowe. Nauczy się szybko. 
LAYER 2 | Bloki 0, 1, 2 | Załadowane| Zamrożone|Conv zamrożone, BN/PReLU odmrożone. 
LAYER 2 | Blok 3 (Ostatni)| Częściowo Losowe| Trenowane |bn2, prelu, bn3 są losowe. 
LAYER 3 | Bloki 0 - 11 | Załadowane| Trenowane|Cały Layer 3 jest odmrożony. 
LAYER 3 | Bloki 12, 13 | Częściowo Losowe| Trenowane | Brakuje bn i prelu. Szybko się nauczą. 
LAYER 4| Wszystkie bloki | Załadowane| Trenowane|Cały Layer 4 jest odmrożony i załadowany. 
OUTPUT | bn2 (Feature BN) | Załadowane| Trenowane|Normalizacja wyjścia. 
OUTPUT | fc (Linear) | Załadowane | Trenowane|Warstwa liniowa przed embeddingiem. 
HEADS | arcface (Nowe) | Losowe | Trenowane|Nowa klasyfikacja tożsamości. 
HEADS |aux_head (Nowe) | Losowe | Trenowane|Wykrywanie maski okluzji. 