# Gas Sensor Array Drift Classification

## Overview

This project investigates multiclass gas classification using Logistic Regression implemented from scratch with NumPy.

The model classifies six different gases from 128 chemical sensor features using a One-vs-Rest (OvR) strategy.

A key focus of the project is evaluating the model under both random and temporal train/test splits to examine how performance changes when the data distribution shifts over time.

## Dataset

The project uses the [Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset) from the UCI Machine Learning Repository.

The dataset contains 13,910 samples with 128 chemical sensor features collected over 10 batches and 36 months.

The task is to classify six different gases:

| Class | Gas |
|------:|-----|
| 1 | Ethanol |
| 2 | Ethylene |
| 3 | Ammonia |
| 4 | Acetaldehyde |
| 5 | Acetone |
| 6 | Toluene |

The dataset contains measurements from 16 chemical sensors, with 128 extracted features describing the sensor responses.

## Problem

The goal of this project is to classify the type of gas from chemical sensor measurements.

A standard random train/test split can provide a useful estimate of classification performance, but it may not reflect how the model performs on data collected at a later time.

Since the dataset was collected over multiple batches and months, this project also evaluates the model on a later batch as unseen future data.

This makes it possible to compare conventional random-split performance with performance under temporal distribution shift.

## Methodology

The dataset was first parsed from the original batch files and combined into a single feature matrix and target vector.

Features were standardized using statistics calculated only from the training data.

A binary Logistic Regression model was implemented from scratch using NumPy, including the sigmoid function, log loss, and gradient descent optimization.

Since the task contains six classes, the binary model was extended to multiclass classification using the One-vs-Rest (OvR) strategy.

The final prediction is selected from the class-specific models based on the highest predicted score.

## Evaluation Strategy

Two evaluation strategies were used to assess the model.

### Random Split

An 80/20 random train/test split was used as the standard evaluation setting.

- Training samples: 11,128
- Test samples: 2,782

### Temporal Split

To evaluate generalization to later data, batches 1–9 were used for training and batch 10 was used as the test set.

- Training samples: 10,310
- Test samples: 3,600

The temporal evaluation was designed to investigate how model performance changes when the test data comes from a later time period.

## Results

The final Logistic Regression model implemented from scratch achieved the following results:

| Evaluation | Accuracy |
|------------|---------:|
| Random Split | 95.72% |
| Temporal Split | 74.44% |

For comparison, a Scikit-learn Logistic Regression model was also evaluated:

| Evaluation | Accuracy |
|------------|---------:|
| Random Split | 99.14% |
| Temporal Split | 71.47% |

The results show a substantial difference between random and temporal evaluation, indicating that performance on randomly sampled test data does not fully represent performance on later batches.

## Error Analysis

The confusion matrices were used to examine class-level prediction errors.

The effect of temporal evaluation was not uniform across the six gas classes.

Acetaldehyde showed the largest recall decrease between the random and temporal evaluations:

| Class | Random Recall | Temporal Recall | Recall Drop |
|-------|--------------:|----------------:|------------:|
| Acetaldehyde | 88.27% | 40.83% | 47.43 pp |

This indicates that some gas classes are more affected by the change in data distribution than others.

## Distribution Shift

Feature distributions were compared between the earlier training batches and the later test batch.

After standardization using the training distribution, measurable shifts were observed across multiple features.

The mean absolute standardized shift was approximately 0.26, while the largest observed feature shift was approximately 1.46.

These changes provide evidence that the feature distributions differ between earlier and later batches. This may be associated with sensor drift and changing measurement conditions, although distribution shift alone does not establish a causal relationship with physical sensor drift.

## Limitations

- The temporal evaluation uses Batch 10 as the future test set.
- Distribution shift does not by itself prove physical sensor drift.
- The current implementation does not include online recalibration or incremental retraining.
- Logistic Regression is a linear model and may not capture complex nonlinear relationships in the sensor data.

## Project Structure

```text
gas-sensor-logistic-regression/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── gas_sensor_classification.ipynb
│
├── src/
│   ├── __init__.py
│   └── logistic_regression.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd gas-sensor-logistic-regression
```

### 2. Create and activate the virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the Gas Sensor Array Drift Dataset and place the batch files inside:

```
data/raw/
```

The directory should contain:

```
batch1.dat
batch2.dat
...
batch10.dat
```

### 5. Run the notebook

```bash
jupyter notebook
```

Then open:

```
notebooks/gas_sensor_classification.ipynb
```

## Future Work

Possible extensions of this project include:

- Periodic model recalibration as new batches become available
- Incremental retraining on newly collected data
- Drift-robust feature engineering
- More detailed statistical drift detection
- Comparison with nonlinear classification models