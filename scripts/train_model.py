from pathlib import Path
from scripts.model import train_from_csv_and_save

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "datasets" / "games.csv"
MODEL_PATH = BASE_DIR / "web" / "model.pkl"

if __name__ == "__main__":
    info = train_from_csv_and_save(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        model_type="LinearRegression",
        n_estimators=800,
        random_state=42
    )

    print("DONE - model trained & saved")
    print(info)
