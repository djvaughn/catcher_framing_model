from logging import INFO, basicConfig, getLogger
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Dict, List, TYPE_CHECKING


if TYPE_CHECKING:
    from polars import LazyFrame

basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


def install_packages():
    """Install required packages if not already available."""
    packages = ["polars", "xgboost"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"]
            )


install_packages()

from polars import Float64, Int64, Series, col, count, min_horizontal, scan_csv
from xgboost import XGBClassifier

PLATE_CENTER_WIDTH_FT = 0.708


def _load_data(training_path: Path) -> LazyFrame:
    """
    Load training data as a Polars LazyFrame.

    Parameters
    ----------
    training_path : Path
        Path to the CSV file containing training data.

    Returns
    -------
    LazyFrame
        Polars LazyFrame for lazy evaluation.
    """
    if not training_path.exists():
        raise FileNotFoundError(f"Input file not found at {training_path}")
    logger.info(f"Loading data from {training_path}")
    return scan_csv(training_path)


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
    logger.info("Engineering features")
    lf = lf.with_columns(
        col("PLATELOCHEIGHT").cast(Float64),
        col("PLATELOCSIDE").cast(Float64),
        col("TOP_ZONE").cast(Float64),
        col("BOT_ZONE").cast(Float64),
        col("BALLS").cast(Int64),
        col("STRIKES").cast(Int64),
        col("VERTAPPRANGLE").cast(Float64),
        col("HORZAPPRANGLE").cast(Float64),
        col("GAME_YEAR").cast(Int64),
    ).with_columns(
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


def _score_model(model_path: Path, audience_lf: LazyFrame) -> LazyFrame:
    """
    Load trained model and generate called strike probabilities.

    Parameters
    ----------
    model_path : Path
        Path to pickled model file.
    audience_lf : LazyFrame
        Feature-engineered pitch data to score.

    Returns
    -------
    LazyFrame
        Input data with CS_PROB column added (probability of called strike).

    Raises
    ------
    FileNotFoundError
        If model file does not exist at specified path.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    with model_path.open("rb") as f:
        model_data: Dict[str, Any] = pickle.load(f)

    model: XGBClassifier = model_data["model"]
    features: List[str] = model_data["feature_columns"]

    logger.info(f"Dropping nulls from {len(features)} features")
    audience_lf = audience_lf.drop_nulls(subset=features + ["IS_STRIKE"])

    X = audience_lf.select(features).collect().to_numpy()
    logger.info(f"Scoring {len(X)} pitches")

    cs_probs = model.predict_proba(X)[:, 1]
    output_lf = audience_lf.with_columns(Series("CS_PROB", cs_probs))
    logger.info(f"Mean CS probability: {cs_probs.mean():.3f}")

    return output_lf


def _pitch_level_data(scored_lf: LazyFrame, pitch_level_csv_path: Path):
    """
    Export pitch-level predictions to CSV.

    Parameters
    ----------
    scored_lf : LazyFrame
        Scored pitch data with CS_PROB column.
    pitch_level_csv_path : Path
        Output path for CSV file.

    Notes
    -----
    Output columns:
        - PITCH_ID: Unique pitch identifier
        - IS_STRIKE: 1 if called strike, 0 otherwise
        - CS_PROB: Model probability of called strike
    """
    logger.info(f"Exporting pitch-level predictions to {pitch_level_csv_path}")
    pitch_level_lf = scored_lf.select(
        [col("PITCH_ID"), col("IS_STRIKE"), col("CS_PROB")]
    )
    pitch_level_lf.collect().write_csv(pitch_level_csv_path)


def _catcher_data(scored_lf: LazyFrame, catcher_csv_path: Path):
    """
    Aggregate framing metrics by catcher and year, export to CSV.

    Parameters
    ----------
    scored_lf : LazyFrame
        Scored pitch data with CS_PROB column.
    catcher_csv_path : Path
        Output path for CSV file.

    Notes
    -----
    Output columns:
        - catcher_id: Unique catcher identifier
        - year: Season year
        - opportunities: Total pitches received
        - actual_cs: Actual called strikes
        - cs_added: Called strikes above expected
        - cs_added_per_100: Called strikes above expected per 100 opportunities
    """
    logger.info("Aggregating catcher metrics")
    catcher_lf = (
        scored_lf.group_by(["CATCHER_ID", "GAME_YEAR"])
        .agg(
            [
                count().alias("opportunities"),
                col("IS_STRIKE").sum().alias("actual_cs"),
                col("CS_PROB").sum().alias("expected_cs"),
            ]
        )
        .with_columns(
            (col("actual_cs") - col("expected_cs")).alias("cs_added"),
        )
        .with_columns(
            (col("cs_added") / col("opportunities") * 100).alias("cs_added_per_100")
        )
    )

    output_lf = catcher_lf.select(
        [
            col("CATCHER_ID").alias("catcher_id"),
            col("GAME_YEAR").alias("year"),
            col("opportunities"),
            col("actual_cs").cast(Int64),
            col("cs_added"),
            col("cs_added_per_100"),
        ]
    )

    output_lf.collect().write_csv(catcher_csv_path)
    logger.info(f"Exported catcher metrics to {catcher_csv_path}")


def main():
    """
    Apply catcher framing model to new data and export predictions.

    Reads new_data.csv, generates predictions, and outputs:
        - pitch_level_predictions.csv: Pitch-level called strike probabilities
        - new_output.csv: Catcher-year aggregated framing metrics
    """
    logger.info("Starting production scoring pipeline")
    model_path = Path("framing_model_prod.pkl")
    data_set_path = Path("new_data.csv")
    pitch_level_csv_path = Path("pitch_level_predictions.csv")
    catcher_csv_path = Path("new_output.csv")

    lf = _load_data(data_set_path)
    audience_lf = _feature_engineering(lf)
    scored_lf = _score_model(model_path, audience_lf)
    _pitch_level_data(scored_lf, pitch_level_csv_path)
    _catcher_data(scored_lf, catcher_csv_path)
    logger.info("Production scoring complete")


if __name__ == "__main__":
    main()
