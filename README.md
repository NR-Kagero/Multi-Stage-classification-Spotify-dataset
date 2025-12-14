# Multi-Stage-classification-Spotify-dataset
# Multi-Layer XGBoost Classifier for Spotify Track Genre Prediction

This Jupyter Notebook implements a hierarchical, two-layer classification system using **XGBoost** to predict the genre of Spotify tracks from their audio features.

The key feature of this notebook is the use of a multi-layer approach to first classify a track into one of 19 arbitrarily defined parent groups, and then use a specific model trained on that group to predict the fine-grained genre.

## Project Overview

* **Task:** Multi-class classification of music tracks into 114 detailed genres.
* **Model:** Two-layer ensemble of XGBoost Classifiers.
* **Dataset:** Spotify Tracks Dataset (113,999 entries).
* **Goal:** Improve fine-grained classification performance by breaking the large classification problem into smaller, specialized sub-problems.

## Data and Preprocessing

1.  **Data Source:** `/kaggle/input/-spotify-tracks-dataset/dataset.csv`.
2.  **Features:** The model uses various audio features like `popularity`, `duration_ms`, `danceability`, `energy`, `loudness`, `acousticness`, `instrumentalness`, `valence`, and `tempo`.
3.  **Target Variables:** The initial target is the `track_genre` (114 unique classes).
4.  **Feature Engineering (Arbitrary Grouping):**
    * The 114 unique genres are arbitrarily grouped into **19 parent categories** (`Arbitrary_Group_01` to `Arbitrary_Group_19`) using an internal function (`create_arbitrary_groups`).
    * A new column, `arbitrary_group`, is created and used for the first layer of classification.
    * All categorical targets (`track_genre` and `arbitrary_group`) are converted to integer factorized labels.
5.  **Splitting:** Data is split into training and test sets (20% test size) on a per-arbitrary-group basis to ensure stratified splits within the overall dataset.

## Methodology: Two-Layer Classification

The classification is performed in two sequential steps:

### Layer 1: Arbitrary Group Prediction (Classifier 1)

* **Model:** A single XGBClassifier is trained on the full dataset.
* **Target:** Predicts the `arbitrary_group` (19 classes).
* **Hyperparameters:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=7`, `gamma=10`, `reg_alpha=0.2`, `reg_lambda=0.5`.

### Layer 2: Fine-Grained Genre Prediction (19 Specialized Classifiers)

* **Models:** 19 separate XGBClassifiers are trained, one for each of the 19 arbitrary groups.
* **Target:** Predicts the original `track_genre` (sub-genres within the group).
* **Hyperparameters:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=7`, `gamma=10`.

### Prediction Flow

For a new track, the full prediction process is:
1.  The Layer 1 model predicts which of the 19 parent groups the track belongs to.
2.  The prediction is passed to the corresponding Layer 2 model for that group, which then outputs the final, specific `track_genre`.

## Results

### Layer 1 Performance (Arbitrary Group Classification)

The model was highly successful at classifying tracks into the 19 arbitrary groups:
* **Accuracy:** 0.9999
* **Macro Avg F1-Score:** 0.9999

### Final Two-Layer Performance (Track Genre Classification - 114 classes)

The complete two-layer system's performance on the original 114-class genre prediction problem:
* **Overall Accuracy:** **0.7842**
* **Macro Avg F1-Score:** 0.7766
* **Weighted Avg F1-Score:** 0.7841
