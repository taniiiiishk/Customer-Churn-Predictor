# Customer-Churn-Predictor

Customer Churn Predictor

A machine learning project that estimates whether a customer is likely to churn. It includes a trained classification model, an interactive browser form, and a REST API built with FastAPI.

The prediction form collects customer profile, account, device, order, engagement, support, and monthly spending details. It returns a churn probability, a `Will churn` or `Will stay` prediction, and a simple risk band.

## Features

- Interactive form served at the site homepage (`/`)
- JSON prediction endpoint at `POST /predict`
- Automatic feature engineering from the customer's monthly spending and account activity
- Saved scikit-learn model loaded from `churn_model.pkl`
- Automatic API documentation at `/docs`
- Jupyter notebook with the data preparation, model training, and evaluation workflow

## Project files

```text
.
├── main.py             # FastAPI app, input schema, feature engineering, and web form
├── churn_model.pkl     # Trained model used by the API
├── large_dataset.csv   # Dataset used for model development
├── ml_pr_04.ipynb      # Exploratory analysis and model training notebook
├── requirements.txt    # Python dependencies
└── Dockerfile
```

## Run locally

Use Python with pip installed. Run these commands from the project directory:

```bash
# Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Start the API and web app
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) to use the prediction form. Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore and call the API.

The app loads `churn_model.pkl` using a path relative to the current working directory, so start Uvicorn from the project directory.

## API

### `POST /predict`

Send a JSON object containing customer details. The request includes fields such as `age`, `tenure_months`, `plan_type`, `support_tickets`, `days_since_last_purchase`, and `monthly_spend`. `monthly_spend` must contain exactly 12 monthly values, ordered oldest to latest. Optional values can be omitted or sent as `null` where accepted by the schema.

Example response:

```json
{
  "churn_probability": 0.7312,
  "prediction": 1,
  "label": "Will churn"
}
```

`prediction` is `1` when the predicted probability is at least 0.5 and `0` otherwise. Use `/docs` to see the complete request schema and submit a sample request.

## Model development

The notebook uses a stratified train/test split and explores preprocessing and classification with scikit-learn, including a tuned Random Forest pipeline. It evaluates models with precision, recall, F1-score, and ROC-AUC in addition to accuracy. The running API uses the pre-trained pipeline in `churn_model.pkl`.

## Dataset and responsible use

The included `large_dataset.csv` is used for model development. Review its source, contents, and any applicable usage terms before redistributing or using it outside this project. Predictions are estimates based on the supplied data; they should support review and analysis rather than be treated as certain outcomes.

## Tech stack

- Python
- FastAPI and Uvicorn
- scikit-learn
- pandas and NumPy
- Pydantic
- joblib
