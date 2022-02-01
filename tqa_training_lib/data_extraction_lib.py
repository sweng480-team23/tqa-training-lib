def extract_data(url: str):
    import pandas as pd

    df = pd.read_json(url)
    return df
