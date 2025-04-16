import pandas as pd
import json
import os
from cleaner import clean_fuel_type


class ValueMapping:
    """
    Třída pro generování mapování hodnot pro dropdowny a validace
    """

    def __init__(self, csv_path):
        """
        Inicializuje mapování ze zvoleného datasetu
        """
        self.mapping = {}
        try:
            self.data = pd.read_csv(csv_path)
            self.clean_data()
        except Exception as e:
            print(f"[ERROR] Chyba při načítání CSV: {e}")
            self.data = pd.DataFrame()

    def clean_data(self):
        """
        Čistí data pro mapování – odstraňuje NaN a normalizuje paliva
        """
        self.data.dropna(subset=['make', 'model', 'fuel_type', 'transmission', 'body_type'], inplace=True)
        self.data['fuel_type'] = self.data['fuel_type'].apply(clean_fuel_type)

    def generate(self):
        """
        Vytvoří mapy pro značky, modely, paliva, převodovky a karoserie
        """
        try:
            self.mapping['makes'] = sorted(self.data['make'].unique())
            self.mapping['models'] = sorted(self.data['model'].unique())
            self.mapping['fuel_types'] = sorted(self.data['fuel_type'].unique())
            self.mapping['transmissions'] = sorted(self.data['transmission'].unique())
            self.mapping['body_types'] = sorted(self.data['body_type'].unique())

            self.mapping['make_model_map'] = self.data.groupby('make')['model'].unique().apply(list).apply(sorted).to_dict()
            self.mapping['model_fuel_types'] = self.data.groupby('model')['fuel_type'].unique().apply(list).apply(sorted).to_dict()
            self.mapping['model_transmissions'] = self.data.groupby('model')['transmission'].unique().apply(list).apply(sorted).to_dict()
            self.mapping['model_body_types'] = self.data.groupby('model')['body_type'].unique().apply(list).apply(sorted).to_dict()
        except Exception as e:
            print(f"[ERROR] Chyba při generování map: {e}")

    def save(self, path=""):
        """
        Uloží výslednou mapu do JSON
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=4)
            print(f"[INFO] Hodnoty byly uloženy do {path}")
        except Exception as e:
            print(f"[ERROR] Chyba při ukládání JSON: {e}")
