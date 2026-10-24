# Cyberbullying Detection on Social Media

A local research prototype that classifies social media posts into the six labels supplied with the dataset. The frontend uses HTML, CSS, and JavaScript; the API uses FastAPI. The model combines word and character TF-IDF features with VADER sentiment scores.

## Dataset and paper

`cyberbullying_tweets.csv` contains 47,692 rows, with `tweet_text` and `cyberbullying_type` columns. The labels are `age`, `ethnicity`, `gender`, `religion`, `other_cyberbullying`, and `not_cyberbullying`. The training process drops 36 repeated exact text/label pairs, while retaining examples whose labels conflict.

The supplied [IEEE Access paper](https://doi.org/10.1109/ACCESS.2025.3558843) describes TF-IDF, sentiment analysis, SMOTE, and several classifiers. It reports 95.38% accuracy and 0.95 F1 for Extra Trees after SMOTE. The paper's description and code/data details are insufficient to reproduce every setting exactly; its prose also alternates between “penta-class” and a six-category dataset. Results below are this repository's measured results and are not directly comparable to the paper's score.

## Training and evaluation

- Text is normalized consistently for training and prediction. Normalized-equivalent texts are assigned to one split group. Only exact repeated text/label pairs are removed; contradictory labels remain in the data and are reported.
- The fixed-seed holdout uses the first three folds of a 10-fold `StratifiedGroupKFold` split, approximately 70:30. It has 33,361 training rows and 14,295 test rows, with zero normalized-text groups shared across partitions.
- Three-fold grouped cross-validation is performed on the training partition. Feature vectorizers are fitted inside each CV fold. Candidate models are selected by cross-validated macro F1; the holdout is evaluated only after selection.
- Candidates include word-only Logistic Regression, word TF-IDF plus VADER Logistic Regression, Linear SVM, word-and-character TF-IDF plus VADER Linear SVM, and Complement Naive Bayes. The best setting is refitted on the full training partition.
- Labels in this dataset are close to balanced. The selected SVM uses class weights; synthetic SMOTE rows are not used for the deployed model. The paper experiment does evaluate SMOTE inside the CV training folds.

### Current holdout results

The baseline and selected model below use the same holdout, grouped split, and preprocessing policy.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Word TF-IDF + VADER · Logistic Regression (baseline) | 83.97% | 0.8410 | 0.8390 | 0.8392 | 0.8400 |
| Word + character TF-IDF + VADER · Linear SVM (selected) | **85.30%** | **0.8520** | **0.8524** | **0.8513** | **0.8520** |

That is a 1.33 percentage-point accuracy increase and a 1.21-point macro F1 increase over the baseline on this holdout. Selected-model three-fold CV macro F1 was 0.8442. The single holdout is an estimate, not proof of improvement across other datasets or platforms.

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| age | 0.9629 | 0.9837 | 0.9732 |
| ethnicity | 0.9751 | 0.9832 | 0.9791 |
| gender | 0.9020 | 0.8687 | 0.8850 |
| not_cyberbullying | 0.6619 | 0.5813 | 0.6190 |
| other_cyberbullying | 0.6530 | 0.7337 | 0.6910 |
| religion | 0.9573 | 0.9637 | 0.9605 |

The largest remaining error is confusion between `not_cyberbullying` and `other_cyberbullying`: 723 examples of the former were predicted as the latter, and 467 in the reverse direction. The source audit found 1,635 normalized-text groups with conflicting labels, covering 3,526 rows. A diagnostic that removed conflicting-label groups only from training changed accuracy to 85.42% and macro F1 to 0.8522; this small change is not used to select or deploy the model.

Training writes the model artifact to `models/cyberbullying_model.joblib` and evaluation outputs to `reports/metrics.json`, `reports/model_comparison.csv`, and confusion-matrix files. These generated artifacts are local and are not required in source control.

## Paper-method experiments

The full bounded comparison has been run with:

```bash
python -m src.paper_experiments
```

It evaluates the paper-style preprocessing approximation, word TF-IDF plus VADER, eight scikit-learn classifiers, and both SMOTE settings. Vectorization and SMOTE are inside each CV pipeline to prevent fold leakage. It uses a stratified 70:30 row split to approximate the paper's split; 824 normalized text groups cross this row split, so these scores are less strict than the grouped holdout above.

| Classifier | Accuracy without SMOTE | Macro F1 without SMOTE | Accuracy with SMOTE | Macro F1 with SMOTE |
|---|---:|---:|---:|---:|
| Logistic Regression | 83.40% | 0.8335 | 83.44% | 0.8339 |
| Random Forest | 81.78% | 0.8187 | 81.63% | 0.8168 |
| Decision Tree | 82.14% | 0.8162 | 82.23% | 0.8166 |
| Naive Bayes | 79.21% | 0.7848 | 79.24% | 0.7852 |
| SVM | 83.91% | 0.8370 | **83.92%** | **0.8370** |
| Gradient Boosting | 74.58% | 0.7438 | 74.60% | 0.7440 |
| Extra Trees | 81.84% | 0.8171 | 82.02% | 0.8193 |
| AdaBoost | 64.57% | 0.6406 | 64.68% | 0.6407 |

The best all-post paper-style setting here is SVM with SMOTE (83.92% accuracy, 0.8370 macro F1). SMOTE changed accuracy by only 0.01 percentage points for SVM. This does not improve on the selected model's grouped holdout metrics; however, the split and preprocessing differ, so direct comparison is approximate. The negative-polarity subset contains 29,723 posts; SVM without SMOTE scored 88.45% accuracy and 0.8408 macro F1 on its own subset holdout. That higher subset accuracy is not full-dataset accuracy and cannot be used to claim an overall improvement.

The paper-filter diagnostic evaluates SVM and Extra Trees on posts with VADER compound polarity below zero. Its subset scores must not be read as full-dataset accuracy.

To run only the negative-polarity SVM/Extra Trees portion, use `python -m src.paper_experiments --filter-only`. A 6,000-row SVM smoke run is available with `python -m src.paper_experiments --smoke`; smoke scores are for execution checks only. Reports are written to `reports/paper_experiments.json` and `.csv` (the smoke run uses separate `_smoke` files).

The paper does not provide all code, split seeds, or parameter grids. This implementation uses Unicode emoji names and Porter stemming as local approximations for the described preprocessing; exact lemmatization resources were unavailable. Gradient Boosting uses a low tree count after a 40-component SVD for tractability. XGBoost is skipped when its dependency is unavailable. See the report metadata for the specific protocol and evaluated models.

## Install and run

Python 3.10 or newer is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data_analysis
python -m src.train_models
```

Start the API in one terminal:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in another:

```bash
python -m http.server 5500 --directory frontend
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500). The API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Available endpoints are `GET /health`, `POST /predict`, and `GET /model-info`. Prediction input is limited to 5,000 characters; submitted posts are not persisted.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"You are such an idiot."}'
```

## Tests

After training the model, run:

```bash
python -m pytest -q
```

The tests cover preprocessing, text-group split rules, experiment pipeline construction, model bundle validation, real-artifact inference, and API request handling.

## Limitations

This is a text-only research prototype. It can miss conversation context, sarcasm, reclaimed terms, and cultural nuance. Conflicting and ambiguous source labels limit the achievable scores, particularly for the two broad bullying categories. The holdout is a single split and is not an external or cross-platform evaluation. Model scores are not calibrated probabilities. Use predictions as review support rather than automatic moderation decisions.
