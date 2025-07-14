import pickle
import pandas as pd
from pathlib import Path

class MemoryVisualizer:
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path

    def load_metadata(self):
        """Loads [(text, ts), …] from the pickle file."""
        with open(self.meta_path, "rb") as f:
            return pickle.load(f)

    def as_dataframe(self):
        """Returns a pandas.DataFrame with columns ['text','timestamp','datetime']."""
        entries = self.load_metadata()
        df = pd.DataFrame(entries, columns=["text","timestamp"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def show(self):
        df = self.as_dataframe()
        # in a Jupyter/Streamlit/Dashboard you’d just render `df`
        print(df.to_markdown(index=False))

if __name__ == "__main__":
    viz = MemoryVisualizer(
        index_path=Path("short_term.index"),
        meta_path=Path("short_term_meta.pkl")
    )
    viz.show()