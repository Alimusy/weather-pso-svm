# PSO Feature Selection with SVM for Weather Prediction

Rainfall prediction comparing a baseline SVM on the full feature space against
an SVM trained on a Binary PSO selected subset.

## Open it

Training and evaluation notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alimusy/weather-pso-svm/blob/main/notebooks/weather_pso_svm.ipynb)

Deploy the app yourself, one click, free:

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Alimusy/weather-pso-svm&branch=main&mainModule=app/app.py)

## Method

1. Median imputation, one-hot encoding and standardisation, all fitted on the
   training partition only, so no evaluation data leaks into preprocessing.
2. Binary PSO encodes feature subsets as binary vectors and scores them with the
   cross-validated accuracy of an internal SVM.
3. Both SVMs trained under identical settings.

## Results

| Model | Features | Accuracy | Recall | Specificity | AUC |
|---|---|---|---|---|---|
| Baseline SVM | 115 | 81.4% | 0.776 | 0.825 | 0.882 |
| PSO + SVM | 47 | 80.9% | 0.765 | 0.821 | 0.873 |

A 59.1% cut in dimensionality for half a point of accuracy, and the PSO model is
markedly cheaper to run: 52.5% less hyperparameter tuning time, 46.6% less
training time.

## Repo layout

```
notebooks/   PSO search, training, evaluation
scripts/     standalone python version of the pipeline
app/         Streamlit app
models/      fitted preprocessor, baseline SVM, PSO-selected SVM, feature indices
figures/     confusion matrices, convergence curve, comparison charts
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset

Rain in Australia (weatherAUS.csv) from Kaggle. Not committed.
