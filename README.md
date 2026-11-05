# Wykrywanie anomalii w dostawach

Projekt zaliczeniowy z eksploracji danych (zajęcia 4–5). Dane są syntetyczne: 1500 dostaw miesięcznie, w każdym miesiącu 45 celowo zmodyfikowanych rekordów (`etykieta = 1`). Etykieta służy wyłącznie do oceny modeli i nigdy nie jest cechą.

## Zawartość

| Plik | Co to jest |
|---|---|
| `anomalie.py` | funkcje: reguły IQR, model normy kosztu, k-means, Isolation Forest, LOF, metryki, koszt, krzywa kosztu |
| `tests/test_anomalie.py` | testy automatyczne (`pytest -q`) |
| `app.py` | dashboard Streamlit |
| `dane/` | `dostawy_styczen.csv` (uczenie), `dostawy_luty.csv` (walidacja), `dostawy_marzec.csv` (nowy cennik od 16.03) |
| `requirements.txt` | biblioteki w wersjach, na których projekt sprawdzono |
| `render.yaml` | konfiguracja wdrożenia na Render |
| `.github/workflows/testy.yml` | testy uruchamiane automatycznie na GitHubie po każdym `git push` |

## 0. Najpierw: uzupełnij `anomalie.py`

W pliku `anomalie.py` dziewięć funkcji ma tylko opis i `raise NotImplementedError` (oznaczenia M1–M5 odpowiadają zadaniom przygotowawczym z karty zadań). Uzupełniaj je po kolei i uruchamiaj `pytest -q`: każdy zielony test to jedna działająca funkcja. Dashboard zadziała, gdy przejdą wszystkie testy.

## 1. Uruchomienie lokalne

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # wszystkie testy powinny przejść
streamlit run app.py      # otworzy się przeglądarka z dashboardem
```

## 2. GitHub

```bash
git init
git add .
git commit -m "Projekt: wykrywanie anomalii w dostawach"
git branch -M main
git remote add origin https://github.com/TWOJ_LOGIN/anomalie-dostawy.git
git push -u origin main
```

Repozytorium zakładasz wcześniej na github.com (New repository, bez README). Po wypchnięciu zakładka **Actions** pokaże wynik testów.

## 3. Render

1. Zaloguj się na render.com kontem GitHub.
2. **New → Blueprint**, wskaż repozytorium: Render odczyta `render.yaml`. Alternatywnie **New → Web Service** i wpisz ręcznie:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Po kilku minutach aplikacja będzie dostępna pod adresem `https://…onrender.com`. W planie bezpłatnym usługa usypia po okresie bezczynności; pierwsze wejście trwa wtedy dłużej.

Nazwy przycisków w GitHubie i Renderze mogą się różnić w zależności od wersji serwisów.

## 4. Co oddać

Adres repozytorium, adres aplikacji na Renderze, notatnik z porównaniem metod oraz w README: wyniki liczbowe (alarmy, precision, recall, F1, koszt) dla budżetu 60 alarmów, trzy wnioski ekonomiczne i jedno ograniczenie analizy.
## Wyniki

Budżet 60 alarmów, styczeń (25 zł za alarm, 800 zł za przeoczoną anomalię):

| Metoda | Alarmy | TP | Precision | Recall | F1 | Koszt |
|---|---|---|---|---|---|---|
| Model normy | 60 | 45 | 0,750 | 1,000 | 0,857 | 1 500 zł |
| LOF | 60 | 37 | 0,617 | 0,822 | 0,705 | 7 900 zł |
| IQR na koszcie | 60 | 22 | 0,367 | 0,489 | 0,419 | 19 900 zł |
| Isolation Forest | 60 | 20 | 0,333 | 0,444 | 0,381 | 21 500 zł |
| K-means | 60 | 17 | 0,283 | 0,378 | 0,324 | 23 900 zł |

Luty, modele, skaler i progi ze stycznia (próg = wynik 60. dostawy w styczniu):

| Metoda | Alarmy | TP | Precision | Recall | F1 | Koszt |
|---|---|---|---|---|---|---|
| Model normy | 57 | 45 | 0,789 | 1,000 | 0,882 | 1 425 zł |
| LOF | 51 | 34 | 0,667 | 0,756 | 0,708 | 10 075 zł |
| IQR na koszcie | 62 | 19 | 0,306 | 0,422 | 0,355 | 22 350 zł |
| Isolation Forest | 64 | 10 | 0,156 | 0,222 | 0,183 | 29 600 zł |
| K-means | 68 | 13 | 0,191 | 0,289 | 0,230 | 27 300 zł |

## Wnioski

1. Przy 60 alarmach miesięcznie model normy kosztuje 1 500 zł, a reguła IQR 19 900 zł: oszczędność 18 400 zł miesięcznie.
2. Bez wiedzy o procesie wybralibyśmy LOF: 7 900 zł w styczniu i 10 075 zł w lutym, czyli ok. 5 do 7 razy więcej niż model normy (1 500 zł i 1 425 zł). Wiedza o tym, od czego zależy koszt dostawy, jest warta więcej niż bardziej zaawansowany algorytm.
3. Po zmianie cennika od 16.03 liczba alarmów rośnie z 4,75 do 15,57 dziennie, a F1 spada do 0,198. Po aktualizacji modelu o stawkę za km (ok. 0,69 zł/km) wraca do 5,5 alarmu dziennie i F1 0,475: potrzebny jest monitor dryfu, a aktualizację powinien zatwierdzać właściciel procesu.

## Ograniczenie analizy

Etykiety obejmują tylko 45 wstrzykniętych błędów miesięcznie w danych syntetycznych. Prawdziwa nietypowa dostawa spoza tej listy (np. zlecenie specjalne) ma etykietę 0, więc jej alarm liczymy jako fałszywy, choć mógł być uzasadniony.

## Aplikacja 