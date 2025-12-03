from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from click import Path as click_path, command, option
from polars import Int64, col, min_horizontal, scan_csv
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from polars import LazyFrame

# CONSTANTS
PLATE_CENTER_WIDTH_FT = 0.708
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


def _load_data(training_path_str: Path) -> LazyFrame:
    """
    Load training data as a Polars LazyFrame.

    Parameters
    ----------
    training_path_str : Path
        Path to the CSV file containing training data.

    Returns
    -------
    LazyFrame
        Polars LazyFrame for lazy evaluation.
    """
    training_data_path = Path(training_path_str)
    return scan_csv(training_data_path)


def _feature_engineering(lf: LazyFrame) -> LazyFrame:
    """
    Engineer features for the catcher framing model.

    Parameters
    ----------
    lf : LazyFrame
        Raw pitch data.

    Returns
    -------
    LazyFrame
        Data with engineered features:
        - IS_STRIKE: 1 if called strike, 0 otherwise
        - IN_ZONE: 1 if pitch is in the strike zone
        - NEAR_VERT_EDGE: Distance to nearest vertical zone edge (feet)
        - NEAR_HORZ_EDGE: Distance to horizontal zone edge (feet)
        - PLATE_LOC_HEIGHT: Pitch height at plate (feet)
        - PLATE_LOC_SIDE: Pitch horizontal location at plate (feet)
        - VERT_APPROACH: Vertical approach angle (degrees)
        - HORZ_APPROACH: Horizontal approach angle (degrees)
    """
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
    """
    Train XGBoost classifier on historical pitching data.

    Parameters
    ----------
    lf : LazyFrame
        Feature-engineered pitch data.

    Returns
    -------
    XGBClassifier
        Trained model for predicting called strike probability.
    """
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


@command()
@option(
    "--input",
    "-i",
    help="Path to the training Data Set",
    type=click_path(exists=True, path_type=Path, dir_okay=False),
    required=True,
)
@option(
    "--output",
    "-o",
    help="Path and file name of where the model pickle file is to go",
    type=click_path(path_type=Path, dir_okay=False, writable=True),
    required=True,
)
def main(input: Path, output: Path):
    """
    Train catcher framing model and save to pickle file.

    Parameters
    ----------
    input : Path
        Path to CSV containing training data.
    output : Path
        Path where model pickle file will be saved.
    """
    lf = _load_data(input)
    lf = _feature_engineering(lf)
    lf = lf.drop_nulls(subset=FEATURES + ["IS_STRIKE"])
    catcher_model = _train_model(lf)
    model_data = {
        "model": catcher_model,
        "feature_columns": FEATURES,
    }
    with output.open("wb") as f:
        pickle.dump(model_data, f)


if __name__ == "__main__":
    main()
