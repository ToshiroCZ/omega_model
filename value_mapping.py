import pandas as pd
import json
import os
from cleaner import clean_fuel_type


class ValueMapping:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.clean_data()
        self.mapping = {}

    def clean_data(self):
        self.data.dropna(subset=['make', 'model', 'fuel_type', 'transmission', 'body_type'], inplace=True)

        # Čistíme fuel_type do standardních kategorií
        self.data['fuel_type'] = self.data['fuel_type'].apply(clean_fuel_type)

    def generate(self):
        self.mapping['makes'] = sorted(self.data['make'].unique())
        self.mapping['models'] = sorted(self.data['model'].unique())
        self.mapping['fuel_types'] = sorted(self.data['fuel_type'].unique())
        self.mapping['transmissions'] = sorted(self.data['transmission'].unique())
        self.mapping['body_types'] = sorted(self.data['body_type'].unique())

        # Make → [Models] map
        make_model_map = self.data.groupby('make')['model'].unique().apply(list).to_dict()
        self.mapping['make_model_map'] = {make: sorted(models) for make, models in make_model_map.items()}

        # Model → [Fuel types] map
        model_fuel_map = self.data.groupby('model')['fuel_type'].unique().apply(list).to_dict()
        self.mapping['model_fuel_types'] = {model: sorted(types) for model, types in model_fuel_map.items()}

        # Model → [Transmissions] map
        model_trans_map = self.data.groupby('model')['transmission'].unique().apply(list).to_dict()
        self.mapping['model_transmissions'] = {model: sorted(types) for model, types in model_trans_map.items()}

        # Model → [Body types] map
        model_body_map = self.data.groupby('model')['body_type'].unique().apply(list).to_dict()
        self.mapping['model_body_types'] = {model: sorted(types) for model, types in model_body_map.items()}

    def save(self, path=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=4)
        print(f"Hodnoty byly uloženy do {path}")
