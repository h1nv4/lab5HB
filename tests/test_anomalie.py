"""Testy funkcji z anomalie.py. Uruchomienie: pytest -q (w katalogu projektu)."""
import os, sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anomalie as A

DANE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dane", "dostawy_styczen.csv")


def test_progi_iqr_na_prostych_liczbach():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])          # Q1 = 2,75; Q3 = 6,25; IQR = 3,5
    lo, hi = A.progi_iqr(x)
    assert lo == pytest.approx(2.75 - 1.5 * 3.5)
    assert hi == pytest.approx(6.25 + 1.5 * 3.5)


def test_alarm_iqr_wskazuje_tylko_wartosc_skrajna():
    x = pd.Series([10, 11, 12, 13, 14, 15, 100])
    assert A.alarm_iqr(x).tolist() == [False] * 6 + [True]


def test_z_odporny_nie_ulega_jednej_wielkiej_wartosci():
    x = pd.Series([10.0, 11, 12, 13, 14, 15, 10_000])
    assert A.z_odporny(x).iloc[-1] > 100            # skrajna wartość daleko
    assert abs(A.z_score(x).iloc[-1]) < 3           # klasyczny z „maskuje” anomalię


def test_metryki_na_malym_przykladzie():
    y = [1, 1, 0, 0, 0]
    a = [1, 0, 1, 0, 0]
    m = A.metryki(y, a)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5) and m["F1"] == pytest.approx(0.5)


def test_koszt_jak_w_przykladzie_z_wykladu():
    # 50 alarmów i 10 przeoczeń: 25 · 50 + 800 · 10 = 9250 zł
    y = np.r_[np.ones(10), np.zeros(50)]
    a = np.r_[np.zeros(10), np.ones(50)]
    assert A.koszt(a, y) == pytest.approx(9250)


def test_alarm_top_zwraca_dokladnie_n_alarmow():
    w = np.array([0.1, 5.0, 0.3, 4.0, 0.2])
    assert A.alarm_top(w, 2).tolist() == [False, True, False, True, False]
    assert A.alarm_top(w, 0).sum() == 0


def test_krzywa_kosztu_zaczyna_od_kosztu_bez_alarmow():
    y = np.array([1, 0, 1, 0])
    c = A.krzywa_kosztu(np.array([0.9, 0.1, 0.8, 0.2]), y)
    assert c.koszt.iloc[0] == pytest.approx(2 * 800)
    assert c.koszt.min() == pytest.approx(2 * 25)       # dwa alarmy łapią obie anomalie


def test_kolumny_kontrolne_nie_moga_byc_cechami():
    df = pd.DataFrame({"koszt_pln": [1.0, 2.0], "etykieta": [0, 1]})
    with pytest.raises(ValueError):
        A.macierz_cech(df, ["koszt_pln", "etykieta"])


def test_model_normy_odroznia_droga_krotka_trase_od_drogiej_dlugiej():
    rng = np.random.default_rng(0)
    n = 300
    km = rng.uniform(50, 800, n); masa = rng.uniform(10, 300, n); eks = rng.integers(0, 2, n)
    data = pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 5, n), unit="D")
    koszt = (45 + 1.75 * km + 2.8 * masa + 180 * eks) * (1 + rng.normal(0, 0.03, n))
    df = pd.DataFrame({"data": data, "odleglosc_km": km, "masa_kg": masa, "ekspres": eks, "koszt_pln": koszt,
                       "opoznienie_h": rng.gamma(1.2, 1.0, n)})
    df.loc[0, ["odleglosc_km", "koszt_pln"]] = [60, 1500]   # krótka trasa, koszt jak dla ~700 km
    df = A.dodaj_cechy(df)
    m = A.model_kosztu(df)
    w = A.wynik_reguly(df, m)
    assert w.iloc[0] > 3.5
    assert w.idxmax() == 0


@pytest.mark.skipif(not os.path.exists(DANE), reason="brak pliku danych")
def test_dane_stycznia_maja_45_anomalii_i_brak_brakow():
    df = A.wczytaj(DANE)
    d = A.diagnostyka(df)
    assert d["wiersze"] == 1500 and d["braki"] == 0 and d["duplikaty_id"] == 0
    assert df["etykieta"].sum() == 45
