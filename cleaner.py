def clean_fuel_type(val):

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
