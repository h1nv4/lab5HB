"""SZKIELET DLA STUDENTÓW. Uzupełnij funkcje z TODO (pytest -q pokaże, co jeszcze nie działa).

Wykrywanie anomalii w dostawach: funkcje używane w notatniku, testach i aplikacji Streamlit.

Konwencja: każda metoda zwraca WYNIK NIETYPOWOŚCI (im wyższy, tym bardziej nietypowa dostawa).
Alarm powstaje dopiero po wyborze progu: reguły (IQR) albo budżetu alarmów (n najwyższych wyników).
Kolumny `etykieta` i `typ_anomalii` służą wyłącznie do oceny; nigdy nie są cechami modelu.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
 
CECHY = ["odleglosc_km", "masa_kg", "ekspres", "czas_h", "opoznienie_h", "koszt_pln"]
KONTROLNE = ["etykieta", "typ_anomalii"]
KOSZT_ALARMU = 25.0       # zł: sprawdzenie jednego alarmu
KOSZT_PRZEOCZENIA = 800.0  # zł: jedna nieznaleziona anomalia
 
 
# ---------------------------------------------------------------- dane
def wczytaj(sciezka: str) -> pd.DataFrame:
    """Wczytuje plik dostaw i ustawia typ daty."""
    df = pd.read_csv(sciezka, parse_dates=["data"])
    df["typ_anomalii"] = df["typ_anomalii"].fillna("")
    return df
 
 
def diagnostyka(df: pd.DataFrame) -> dict:
    """Podstawowe kontrole jakości przed modelowaniem."""
    return {
        "wiersze": len(df),
        "braki": int(df[CECHY].isna().sum().sum()),
        "duplikaty_id": int(df["id_dostawy"].duplicated().sum()),
        "duplikaty_wierszy": int(df.drop(columns=["id_dostawy"]).duplicated().sum()),
        "ujemne": int((df[CECHY] < 0).sum().sum()),
        "udzial_ekspres": float(df["ekspres"].mean()),
    }
 
 
# ---------------------------------------------------------------- reguły jednowymiarowe
def progi_iqr(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Granice Q1 − k·IQR i Q3 + k·IQR (kwantyle liniowe, jak w pandas)."""
    x = pd.Series(x)
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)
 
 
def alarm_iqr(x: pd.Series, k: float = 1.5, tylko_gorny: bool = True) -> pd.Series:
    """True, gdy wartość wychodzi poza granice IQR (domyślnie tylko górną)."""
    x = pd.Series(x)
    dolny, gorny = progi_iqr(x, k)
    if tylko_gorny:
        return x > gorny
    return (x < dolny) | (x > gorny)
 
 
def z_score(x: pd.Series) -> pd.Series:
    """Klasyczny z = (x − średnia) / odchylenie standardowe (próby)."""
    return (x - x.mean()) / x.std()
 
 
def z_odporny(x: pd.Series) -> pd.Series:
    """Odporny z: (x − mediana) / (1,4826 · MAD). Mało wrażliwy na same anomalie."""
    x = pd.Series(x)
    med = x.median()
    skala = 1.4826 * (x - med).abs().median()
    return (x - med) / skala
 
 
# ---------------------------------------------------------------- model normy
CECHY_MODELU = ["odleglosc_km", "masa_kg", "ekspres", "sobota"]
 
 
def dodaj_cechy(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy kontekstu: sobota (dopłata weekendowa) i opóźnienie względem mediany dnia (pogoda, korki)."""
    out = df.copy()
    out["sobota"] = (out["data"].dt.weekday == 5).astype(int)
    out["opoznienie_wzgl_dnia"] = out["opoznienie_h"] - out.groupby("data")["opoznienie_h"].transform("median")
    return out
 
 
def model_kosztu(df: pd.DataFrame, cechy: list[str] | None = None) -> LinearRegression:
    """Regresja liniowa kosztu na cechach trasy i usługi: „model normy” kosztu."""
    cechy = cechy or CECHY_MODELU
    return LinearRegression().fit(df[cechy], df["koszt_pln"])
 
 
def reszta_kosztu(df: pd.DataFrame, model: LinearRegression) -> pd.Series:
    """Koszt rzeczywisty minus koszt oczekiwany dla tej trasy, masy, usługi i dnia tygodnia."""
    cechy = list(model.feature_names_in_)
    oczekiwany = model.predict(df[cechy])
    return pd.Series(df["koszt_pln"].to_numpy() - oczekiwany, index=df.index)
 
 
def parametry_normy(x: pd.Series) -> tuple[float, float]:
    """Mediana i 1,4826 · MAD: środek i skala „normalnych” wartości."""
    med = float(x.median())
    return med, float(1.4826 * (x - med).abs().median())
 
 
def wynik_reguly(df: pd.DataFrame, model: LinearRegression, norma: dict | None = None) -> pd.Series:
    """Wynik reguł z kontekstem: większy z odpornych z dla reszty kosztu i opóźnienia względem dnia.
    `norma` (mediany i skale ze zbioru uczącego) pozwala ocenić nowy miesiąc tą samą miarą."""
    if "opoznienie_wzgl_dnia" not in df.columns or "sobota" not in df.columns:
        df = dodaj_cechy(df)
    reszta = reszta_kosztu(df, model)
    opoznienie = df["opoznienie_wzgl_dnia"]
    if norma is None:
        norma = {"reszta": parametry_normy(reszta), "opoznienie": parametry_normy(opoznienie)}
    med_r, skala_r = norma["reszta"]
    med_o, skala_o = norma["opoznienie"]
    z_reszta = (reszta - med_r) / skala_r
    z_opoznienie = (opoznienie - med_o) / skala_o
    return pd.Series(np.maximum(z_reszta, z_opoznienie), index=df.index)
 
 
def norma_reguly(df: pd.DataFrame, model: LinearRegression) -> dict:
    """Parametry normy wyznaczone na danych uczących (do użycia w kolejnych miesiącach)."""
    return {"reszta": parametry_normy(reszta_kosztu(df, model)), "opoznienie": parametry_normy(df["opoznienie_wzgl_dnia"])}
 
 
# ---------------------------------------------------------------- metody wielowymiarowe
def macierz_cech(df: pd.DataFrame, cechy: list[str] | None = None, skaler: StandardScaler | None = None):
    """Standaryzuje cechy. Zwraca (macierz, dopasowany skaler). Kolumn kontrolnych nie wolno podać."""
    cechy = cechy or CECHY
    if set(cechy) & set(KONTROLNE + ["id_dostawy", "data"]):
        raise ValueError("Kolumny kontrolne i identyfikatory nie mogą być cechami modelu")
    skaler = skaler or StandardScaler().fit(df[cechy])
    return skaler.transform(df[cechy]), skaler
 
 
def wynik_kmeans(X: np.ndarray, k: int = 4, seed: int = 0, model: KMeans | None = None):
    """Odległość do najbliższego centroidu. Zwraca (wynik, model)."""
    model = model or KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return model.transform(X).min(axis=1), model
 
 
def wynik_isolation_forest(X: np.ndarray, seed: int = 0, n_drzew: int = 300, model: IsolationForest | None = None):
    """Minus score_samples: wyższy = bardziej nietypowy. Zwraca (wynik, model)."""
    model = model or IsolationForest(n_estimators=n_drzew, random_state=seed).fit(X)
    return -model.score_samples(X), model
 
 
def wynik_lof(X: np.ndarray, n_sasiadow: int = 20, X_uczenie: np.ndarray | None = None):
    """Wynik LOF (≈ 1 typowy, > 1 lokalnie odstający).
    Bez X_uczenie: wynik na tych samych danych (negative_outlier_factor_).
    Z X_uczenie: model w trybie novelty, ocena nowych danych."""
    if X_uczenie is None:
        lof = LocalOutlierFactor(n_neighbors=n_sasiadow).fit(X)
        return -lof.negative_outlier_factor_
    lof = LocalOutlierFactor(n_neighbors=n_sasiadow, novelty=True).fit(X_uczenie)
    return -lof.score_samples(X)
 
 
# ---------------------------------------------------------------- progi, ocena, koszt
def alarm_top(wynik, n: int) -> np.ndarray:
    """Alarm dla n dostaw o najwyższym wyniku (budżet alarmów)."""
    w = np.asarray(wynik, dtype=float)
    alarm = np.zeros(len(w), dtype=bool)
    if n > 0:
        alarm[np.argsort(-w, kind="stable")[:n]] = True
    return alarm
 
 
def alarm_prog(wynik, prog: float) -> np.ndarray:
    """Alarm, gdy wynik przekracza ustalony próg (np. wyznaczony na danych uczących)."""
    return np.asarray(wynik) > prog
 
 
def metryki(y, alarm) -> dict:
    """Macierz pomyłek, precision, recall, F1 i koszt operacyjny."""
    y = np.asarray(y).astype(bool)
    a = np.asarray(alarm).astype(bool)
    tp = int((y & a).sum())
    fp = int((~y & a).sum())
    fn = int((y & ~a).sum())
    tn = int((~y & ~a).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"alarmy": int(a.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": p, "recall": r, "F1": f1,
            "koszt": koszt(a, y)}
 
 
def koszt(alarm, y, c_alarm: float = KOSZT_ALARMU, c_fn: float = KOSZT_PRZEOCZENIA) -> float:
    """Koszt = c_alarm · liczba alarmów + c_fn · liczba przeoczonych anomalii."""
    y = np.asarray(y).astype(bool)
    a = np.asarray(alarm).astype(bool)
    fn = int((y & ~a).sum())
    return float(c_alarm * a.sum() + c_fn * fn)
 
 
def krzywa_kosztu(wynik, y, n_max: int | None = None, c_alarm: float = KOSZT_ALARMU, c_fn: float = KOSZT_PRZEOCZENIA) -> pd.DataFrame:
    """Koszt dla budżetu n = 0..n_max alarmów (alarm dla n najwyższych wyników)."""
    w = np.asarray(wynik, dtype=float)
    y = np.asarray(y).astype(int)
    n_max = len(w) if n_max is None else min(n_max, len(w))
    y_posortowane = y[np.argsort(-w, kind="stable")]
    tp = np.concatenate([[0], np.cumsum(y_posortowane)])[: n_max + 1]
    n = np.arange(n_max + 1)
    fn = y.sum() - tp
    return pd.DataFrame({"alarmy": n, "TP": tp, "FN": fn,
                         "koszt": c_alarm * n + c_fn * fn})
 
 
def recall_wg_typu(df: pd.DataFrame, alarm) -> pd.Series:
    """Jaki odsetek anomalii każdego typu dostał alarm."""
    a = pd.Series(np.asarray(alarm).astype(bool), index=df.index)
    t = df["typ_anomalii"]
    return a[t != ""].groupby(t[t != ""]).mean()
 
 
def ranking(df: pd.DataFrame, wyniki: dict[str, np.ndarray], n: int = 20) -> pd.DataFrame:
    """Tabela dostaw z wynikami metod i rangą średnią (1 = najbardziej nietypowa)."""
    out = df[["id_dostawy", "data"] + CECHY].copy()
    for k, v in wyniki.items():
        out[k] = v
    rangi = pd.DataFrame({k: pd.Series(v).rank(ascending=False).to_numpy() for k, v in wyniki.items()}, index=df.index)
    out["ranga_srednia"] = rangi.mean(axis=1)
    return out.sort_values("ranga_srednia").head(n)
 
