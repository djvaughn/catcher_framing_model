from typing import TYPE_CHECKING
from pathlib import Path
import pickle

from polars import Int64, col, min_horizontal, scan_csv
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from polars import LazyFrame


PLATE_CENTER_WIDTH_FT = 0.708


def _load_data(training_path_str: str) -> LazyFrame:
    training_data_path = Path(training_path_str)
    return scan_csv(training_data_path)


FEATURES = [
    "PLATE_LOC_HEIGHT",
    "PLATE_LOC_SIDE",
    "TOP_ZONE",
    "BOT_ZONE",
    "IN_ZONE",
    "NEAR_VERT_EDGE",
    "NEAR_HORZ_EDGE",
    "BALLS",
    "STRIKES",
    "VERT_APPROACH",
    "HORZ_APPROACH",
]

GAME_YEARS = [2021, 2022]


def _feature_engineering(lf: LazyFrame) -> LazyFrame:
    lf = lf.with_columns(
        IS_STRIKE=(col("PITCHCALL") == "StrikeCalled").cast(Int64),
        IN_ZONE=(
            (col("BOT_ZONE") <= col("PLATELOCHEIGHT"))
            & (col("PLATELOCHEIGHT") <= col("TOP_ZONE"))
        ).cast(Int64),
        NEAR_VERT_EDGE=min_horizontal(
            (col("PLATELOCHEIGHT") - col("BOT_ZONE")).abs(),
            (col("PLATELOCHEIGHT") - col("TOP_ZONE")).abs(),
        ),
        NEAR_HORZ_EDGE=(col("PLATELOCSIDE") - PLATE_CENTER_WIDTH_FT).abs(),
        PLATE_LOC_HEIGHT=col("PLATELOCHEIGHT"),
        PLATE_LOC_SIDE=col("PLATELOCSIDE"),
        VERT_APPROACH=col("VERTAPPRANGLE"),
        HORZ_APPROACH=col("HORZAPPRANGLE"),
    )
    return lf


def _train_model(lf: LazyFrame) -> XGBClassifier:
    train_lf = lf.filter(col("GAME_YEAR").is_in(GAME_YEARS))
    X_train = train_lf.select(FEATURES).collect().to_numpy()
    y_train = train_lf.select("IS_STRIKE").collect().to_numpy().ravel()

    model = XGBClassifier(
        objective="binary:logistic",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=100,
    )
    model.fit(X_train, y_train)
    return model


def main():
    lf = _load_data("ML_TAKES_ENCODED.csv")
    lf = _feature_engineering(lf)
    lf = lf.drop_nulls(subset=FEATURES + ["IS_STRIKE"])
    catcher_model = _train_model(lf)
    model_path = Path("framing_model.pkl")
    model_data = {
        "model": catcher_model,
        "feature_columns": FEATURES,
    }
    with model_path.open("wb") as f:
        pickle.dump(model_data, f)


if __name__ == "__main__":
    main()
