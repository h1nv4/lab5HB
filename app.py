"""Dashboard: wykrywanie anomalii w dostawach. Uruchomienie lokalne: streamlit run app.py"""
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import anomalie as A

KATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dane")
PLIKI = {"Styczeń 2026 (uczenie)": "dostawy_styczen.csv", "Luty 2026 (walidacja)": "dostawy_luty.csv", "Marzec 2026 (nowy cennik od 16.03)": "dostawy_marzec.csv"}
METODY = ["Model normy", "Isolation Forest", "LOF", "K-means", "IQR koszt"]

st.set_page_config(page_title="Anomalie w dostawach", layout="wide")


@st.cache_resource
def modele_uczone():
    """Wszystkie modele i skaler uczymy raz, na styczniu. Inne miesiące oceniamy tymi samymi modelami."""
    uc = A.dodaj_cechy(A.wczytaj(os.path.join(KATALOG, PLIKI["Styczeń 2026 (uczenie)"])))
    mk = A.model_kosztu(uc)
    X, sk = A.macierz_cech(uc)
    return {"uczenie": uc, "model_kosztu": mk, "norma": A.norma_reguly(uc, mk), "X": X, "skaler": sk,
            "kmeans": A.wynik_kmeans(X, 5)[1], "if": A.wynik_isolation_forest(X)[1]}


def wynik(metoda: str, df: pd.DataFrame, M: dict) -> np.ndarray:
    if metoda == "Model normy":
        return A.wynik_reguly(df, M["model_kosztu"], M["norma"]).to_numpy()
    if metoda == "IQR koszt":
        return df["koszt_pln"].to_numpy()
    X, _ = A.macierz_cech(df, skaler=M["skaler"])
    if metoda == "Isolation Forest":
        return A.wynik_isolation_forest(X, model=M["if"])[0]
    if metoda == "K-means":
        return A.wynik_kmeans(X, model=M["kmeans"])[0]
    return A.wynik_lof(X, 20, X_uczenie=M["X"])


M = modele_uczone()
st.title("Wykrywanie anomalii w dostawach")
st.caption("Dane syntetyczne do zajęć z eksploracji danych. Modele uczone na styczniu 2026 r.; alarm = dostawy o najwyższym wyniku nietypowości.")

with st.sidebar:
    st.header("Ustawienia")
    zrodlo = st.selectbox("Miesiąc", list(PLIKI), index=0)
    wlasny = st.file_uploader("albo własny plik CSV (te same kolumny)", type="csv")
    metoda = st.selectbox("Metoda", METODY, index=0)
    budzet = st.slider("Budżet alarmów (ile dostaw sprawdzamy)", 5, 300, 60, step=5)
    c_alarm = st.number_input("Koszt sprawdzenia alarmu [zł]", 0.0, 1000.0, A.KOSZT_ALARMU, step=5.0)
    c_fn = st.number_input("Koszt przeoczenia anomalii [zł]", 0.0, 10000.0, A.KOSZT_PRZEOCZENIA, step=50.0)

if wlasny is not None:
    df = pd.read_csv(wlasny, parse_dates=["data"])
    if "typ_anomalii" not in df:
        df["typ_anomalii"] = ""
    df["typ_anomalii"] = df["typ_anomalii"].fillna("")
else:
    df = A.wczytaj(os.path.join(KATALOG, PLIKI[zrodlo]))
df = A.dodaj_cechy(df)
w = wynik(metoda, df, M)
alarm = A.alarm_top(w, budzet)
df["wynik"] = w
df["alarm"] = alarm

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Dostawy", f"{len(df):,}".replace(",", " "))
k2.metric("Alarmy", int(alarm.sum()))
if "etykieta" in df:
    m = A.metryki(df["etykieta"], alarm)
    k3.metric("Precision", f"{m['precision']:.2f}")
    k4.metric("Recall", f"{m['recall']:.2f}")
    k5.metric("F1", f"{m['F1']:.2f}")
    k6.metric("Koszt [zł]", f"{A.koszt(alarm, df['etykieta'], c_alarm, c_fn):,.0f}".replace(",", " "))
else:
    k3.info("Brak kolumny etykieta: metryk nie da się policzyć.")

lewa, prawa = st.columns(2)
with lewa:
    st.subheader("Koszt względem odległości")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(df.loc[~alarm, "odleglosc_km"], df.loc[~alarm, "koszt_pln"], s=8, color="#A9D0D4", label="bez alarmu")
    ax.scatter(df.loc[alarm, "odleglosc_km"], df.loc[alarm, "koszt_pln"], s=28, color="#C4532F", label="alarm")
    ax.set_xlabel("odległość [km]"); ax.set_ylabel("koszt [zł]"); ax.legend(frameon=False)
    st.pyplot(fig)
with prawa:
    if "etykieta" in df:
        st.subheader("Koszt a liczba alarmów")
        c = A.krzywa_kosztu(w, df["etykieta"], n_max=400, c_alarm=c_alarm, c_fn=c_fn)
        st.line_chart(c.set_index("alarmy")["koszt"])
        opt = c.loc[c.koszt.idxmin()]
        st.caption(f"Najniższy koszt: {opt.koszt:,.0f} zł przy {int(opt.alarmy)} alarmach.".replace(",", " "))

st.subheader("Ranking najbardziej nietypowych dostaw")
kol = ["id_dostawy", "data", "odleglosc_km", "masa_kg", "ekspres", "opoznienie_h", "koszt_pln", "wynik"] + (["etykieta", "typ_anomalii"] if "etykieta" in df else [])
top = df.sort_values("wynik", ascending=False).head(budzet)[kol].assign(data=lambda t: t["data"].dt.strftime("%Y-%m-%d"), wynik=lambda t: t["wynik"].round(3))
st.dataframe(top, width="stretch", hide_index=True)
st.download_button("Pobierz alarmy (CSV)", top.to_csv(index=False).encode("utf-8"), file_name=f"alarmy_{metoda.replace(' ', '_')}.csv", mime="text/csv")
