# Insurance Claims Fraud Review

This project explores how a claims team could use a machine learning model to prioritise insurance claims for manual review. The Streamlit demo lets you compare what happens when the team can review 10%, 20%, or 30% of incoming claims.

## Run the demo

From a terminal in this folder, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal. The app reads three precomputed validation scenarios from `validation_scenarios.csv` and displays a fixed evaluation on later claims.

## Model and evaluation

I joined five coursework data files into 28,836 labelled claims and compared Logistic Regression and Random Forest using 13 features. I split the data chronologically into training, validation, and test sets. Random Forest achieved a validation average precision of 0.614, compared with 0.395 for Logistic Regression.

I selected a score cutoff on the validation set to refer up to 20% of claims for review. I then applied that same cutoff to 5,496 later claims.

| Metric | Validation | Held-out test |
|---|---:|---:|
| Claims referred for review | 20.0% | 19.5% |
| Fraud recall | 46.0% | 30.3% |
| Referral precision | 63.5% | 32.3% |
| Average precision | 0.614 | 0.340 |

The review workload stayed close to 20% on later claims, while the model caught a smaller share of labelled fraud. This is an internal evaluation on a later portion of the same coursework dataset.

## Files

- `app.py` — Streamlit demo for exploring review capacity
- `validation_scenarios.csv` — Aggregate validation results for the three slider settings
- `requirements.txt` — Python dependencies for running the demo
- `insurance_fraud_portfolio.ipynb` — Model training and evaluation notebook

The Streamlit demo runs with the included summary CSV. To rerun the training notebook, place the five original coursework CSVs in a local `data/` folder beside the notebook. The original claim records are kept separately.