pobierz dataset KAGGLE_DATASET tu jest nazwa w zmiennej env
zapisz na dysku jesli nie unzipped to unzip
w plikach jest struktura: webface-112x112 folder i w nim foldery o nazwie id_0 - id_10571 np no
w tych folderach zdjęcia
pierwsze musimy pobrac nazwy wszystkich folderów w folderze webface-112x112 i zrobic shuffle 
podzielic w proporcji 8:1:1 na train:val:test
przeniesc foldery z test do folderu test/
przeniesc foldery z val do folderu val/
przeniesc foldery z train do folderu train/
dla każdego zdjęcia odpalić Detektor twarzy z retinaFace i wywołać aby zapisać landmarks czyli bbox twarzy i gdzie oczy, usta, nos
zapisac jak mielismy 0.jpg no to folder 0 i w nim zdjecie 0.jpg i 0.json
tak dla kazdego folderu zaczynajac od test potem val i train
jak bedziemy juz miec komplet to trzeba caly folder webface-112x112 wyslac za pomoca gsutil -m do bucketu o nazwie BUCKET_NAME
