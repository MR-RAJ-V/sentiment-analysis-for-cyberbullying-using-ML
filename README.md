# Cyberbullying Detection on Social Media

A local, full-stack research prototype that combines TF-IDF text features and VADER sentiment features to classify social media posts into the labels supplied with the dataset. The frontend uses plain HTML, CSS, and JavaScript; the API uses FastAPI.

## Input files and findings

The supplied CSV is `cyberbullying_tweets.csv`. It has 47,692 rows and two columns: `tweet_text` and `cyberbullying_type`. It contains six labels: `age`, `ethnicity`, `gender`, `religion`, `other_cyberbullying`, and `not_cyberbullying`. There are no missing values and 36 exact duplicate rows; training removes exact duplicate text/label pairs. Class counts are close, from 7,823 to 7,998 rows. A further audit found 1,675 repeated-text rows, and 1,639 repeated-text groups have conflicting labels. This disagreement is a limitation in the source labels.

The supplied IEEE Access paper (DOI [10.1109/ACCESS.2025.3558843](https://doi.org/10.1109/ACCESS.2025.3558843)) describes a 70:30 split, TF-IDF, VADER-style sentiment analysis, SMOTE, and several classifiers. It reports 95.38% accuracy and 0.95 F1 for Extra Trees after SMOTE. Its prose alternates between “penta-class” and a six-category dataset and describes a strict nonnegative sentiment filter that can discard harmful posts. This project keeps all posts for the deployed model and reports that filter as a separate negative-polarity subset diagnostic.

## Method

- A fixed-seed, stratified 70:30 group split is made before fitting TF-IDF. Identical source texts are kept entirely in train or test, including repeated texts that have conflicting labels.
- Text normalization is shared by training and prediction. It normalizes case, HTML, links, mentions, and whitespace and uses NLTK’s tweet tokenizer while retaining negation, hashtags, emoji, and expressive punctuation.
- TF-IDF uses word unigrams and bigrams. Four VADER scores (negative, neutral, positive, compound) are appended as sparse features.
- Experiments compare TF-IDF-only Logistic Regression against TF-IDF + VADER Logistic Regression, Linear SVM, Complement Naive Bayes, and Extra Trees.
- The deployed model is Logistic Regression with TF-IDF + VADER. This is fixed before holdout evaluation so the test set is not used to choose the model. It exposes class probabilities; they are not calibrated probabilities.
- SMOTE is not used: the six classes are already close to balanced, and interpolating sparse word vectors would add synthetic feature combinations. A class-weighted model is used instead.
- The paper’s strict negative-polarity approach is shown only as a diagnostic on the filtered holdout subset. That subset result is selection-biased and is not directly comparable with full-holdout metrics.

## Project layout

```text
backend/             FastAPI application and request/response schemas
frontend/            Responsive HTML/CSS/vanilla JS dashboard
src/                 Data analysis, preprocessing, features, training, inference
tests/               Preprocessing, persistence, inference, and API tests
models/               Trained Joblib artifact (created by training)
reports/              Data quality, charts, metrics, and confusion matrix
cyberbullying_tweets.csv  Supplied source data (unchanged)
Integrating_Sentiment_Analysis_With_Machine_Learning_for_Cyberbullying_Detection_on_Social_Media.pdf  Supplied paper (unchanged)
```

## Install and train

Python 3.10 or newer is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data_analysis
python -m src.train_models
```

Training writes `models/cyberbullying_model.joblib` and reports in `reports/`. Training fits the vectorizer only on the training partition and never resamples the test partition. The paper’s XGBoost model is not included in the local comparison: its current package attempted to add a large NVIDIA NCCL dependency in this CPU environment; four scikit-learn models are compared instead.

### Local evaluation results

These are measured results from this project’s fixed-seed, 33,358-post training set and 14,298-post stratified group holdout. Exact source texts do not cross partitions. They are separate from the IEEE paper’s reported scores.

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF only · Logistic Regression | 84.28% | 0.8424 | 0.8431 |
| TF-IDF + VADER · Logistic Regression (deployed) | 84.32% | 0.8429 | 0.8437 |
| TF-IDF + VADER · Linear SVM | 83.97% | 0.8379 | 0.8386 |
| TF-IDF only · Complement NB | 77.70% | 0.7592 | 0.7599 |
| TF-IDF + VADER · Extra Trees | 80.54% | 0.8045 | 0.8053 |

The dataset has substantial label disagreement for repeated text: 1,639 repeated-text groups have multiple target labels. The model also confuses `not_cyberbullying` with `other_cyberbullying` more often than the other categories. Treat this as a limitation of the data and use the predictions as review support.

## Run the app

In terminal 1, with the virtual environment active:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

In terminal 2:

```bash
python -m http.server 5500 --directory frontend
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500). The API docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The frontend calls the local API at `http://127.0.0.1:8000`.

### API example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"You are such an idiot."}'
```

Endpoints are `GET /health`, `POST /predict`, and `GET /model-info`. Text is limited to 5,000 characters. Submitted posts are not persisted.

## Tests

After training the model, run:

```bash
pytest -q
```

The API test suite uses HTTPX ASGI transport. FastAPI’s synchronous `TestClient` startup hung in this runtime even with an empty FastAPI app; live HTTP and browser integration were checked separately against the actual local servers.

## Limitations

This is a text-only research prototype. It may miss conversational context, sarcasm, reclaimed terms, and cultural nuance. Dataset labels reflect the dataset’s category scheme and should not be treated as definitive findings about a real interaction. The model probability is not calibrated. The reported local holdout metrics are a single split, not an external or cross-platform evaluation. Use predictions as review support, not as an automatic moderation decision.
