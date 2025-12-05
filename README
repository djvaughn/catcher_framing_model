# Catcher Framing Model

A machine learning model to evaluate MLB catcher framing skill using XGBoost.

## Requirements

- Python 3.14.0
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── model/
│   ├── feature_eng.ipynb         # Jupyter Notebook of Model Training
│   ├── train_model.py            # Training script
│   └── feature_eng.pdf           # Model methodology and validation
├── framing_model_prod.pkl        # Trained model
├── production.py                 # Production scoring script
├── requirements.txt              # Python dependencies
└── README.md
```

## Usage

### Training

```bash
python model/train_model.py --input <training_data.csv> --output model/framing_model_prod.pkl
```

### Production Scoring

Place `new_data.csv` and `framing_model_prod.pkl` in the same directory as `production.py`, then run:

```bash
python production.py
```

This outputs:
- `pitch_level_predictions.csv` - Pitch-level called strike probabilities
- `new_output.csv` - Catcher-year aggregated framing metrics

## Model Documentation

See `model/model_documentation.pdf` for details on feature engineering, model training, and validation.
