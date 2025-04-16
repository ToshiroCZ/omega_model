import unicodedata
import pandas as pd


def clean_fuel_type(val):
    """
    Standardizuje typy paliv na jednotné názvy.
    """
    if not isinstance(val, str):
        return None
    if "Battery EV" in val:
        return "Battery EV (BEV)"
    elif "Full-Hybrid" in val:
        return "Full-Hybrid (FHEV)"
    elif "Plug-in Hybrid" in val:
        return "Plug-in Hybrid (PHEV)"
    elif "Mild-Hybrid" in val:
        return "Mild-Hybrid (MHEV)"
    elif "Benzín" in val:
        return "Benzín"
    elif "Diesel" in val:
        return "Diesel"
    elif "LPG" in val:
        return "LPG"
    elif "CNG" in val:
        return "CNG"
    return val


def normalize_text(text):
    """
    Odstraní diakritiku, převede na TitleCase.
    Např. "mercedes-benz" -> "Mercedes-Benz"
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower().title()


def merge_and_clean_datasets(aaa_path, esa_path, output_path):
    """
    Spojí a vyčistí dataset z AAA a AutoESA a uloží ho jako jeden soubor.
    """
    try:
        print("[INFO] Merging datasets...")
        aaa_df = pd.read_csv(aaa_path)
        esa_df = pd.read_csv(esa_path)

        if "car_id" not in esa_df.columns:
            esa_df.columns = aaa_df.columns

        df = pd.concat([aaa_df, esa_df], ignore_index=True)
        cleaned = clean_dataset(df)
        cleaned.to_csv(output_path, index=False)
        print(f"[INFO] Combined dataset saved as {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to merge datasets: {e}")


def clean_dataset(df):
    """
    Provede kompletní čištění již existujícího DataFrame:
    - normalizace značek/modelů
    - čištění typu paliva
    - převod číselných hodnot
    - odstranění nevalidních řádků
    """
    try:
        df["make"] = df["make"].apply(normalize_text)
        df["model"] = df["model"].apply(normalize_text)
        df["fuel_type"] = df["fuel_type"].apply(clean_fuel_type)

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
        df["engine_power"] = pd.to_numeric(df["engine_power"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df.dropna(subset=[
            "price", "make", "model", "fuel_type", "transmission", "body_type",
            "year", "mileage", "engine_power"
        ], inplace=True)
        df = df[df["price"] > 0]

        return df
    except Exception as e:
        print(f"[ERROR] Failed to clean dataset: {e}")
        return pd.DataFrame()
