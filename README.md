# ML.NET Taxi Fare Prediction Exercise

A hands-on **Predictive AI and ML.NET regression exercise** demonstrating how **data quality, feature engineering, model selection, and error analysis** affect predictive model performance.

This project uses a taxi fare dataset with more than **1 million observations** and builds several ML.NET regression models to answer a simple business question:

> **Given what we know about a taxi trip, how accurately can we predict the fare?**

The goal is not simply to build the highest-scoring model. The exercise is designed to demonstrate the experimental process behind practical predictive AI.

---
## Get more on this exercise:
https://aindotnet.com/2026/08/mlnet-predictive-ai-taxi-fare-exercise/

## Get more resources on Predictive AI:
https://aindotnet.com/forecasting/
---

## What You Will Learn

This exercise demonstrates several important machine-learning concepts:

* Establishing a baseline model
* Separating training and test data
* Evaluating regression models
* Understanding R-Squared, MAE, RMSE, and MSE
* Identifying poor predictions and outliers
* Profiling data before modeling
* Measuring the effect of data cleaning
* Measuring the effect of feature engineering
* Comparing machine-learning algorithms
* Keeping experimental variables controlled
* Understanding why better data and features can matter more than changing algorithms

---

# The Experiment

Rather than training one model, the application runs six controlled experiments.

| Run   | Data     | Features | Algorithm  | Purpose                                   |
| ----- | -------- | -------- | ---------- | ----------------------------------------- |
| **A** | Original | Basic    | FastTree   | Establish the baseline                    |
| **B** | Original | Full     | FastTree   | Measure the effect of additional features |
| **C** | Clean    | Basic    | FastTree   | Measure the effect of basic data cleaning |
| **D** | Clean    | Full     | FastTree   | Combine improved data and features        |
| **E** | Clean    | Full     | LightGBM   | Compare algorithms                        |
| **F** | Clean    | Full     | FastForest | Compare another tree algorithm            |

The same train/test split is reused when comparing related experiments so that differences are caused by the experimental change rather than by different random samples.

---

# The "Ah-Hah" Moments

## Ah-Hah #1 — A Good Aggregate Score Can Hide Bad Predictions

The initial FastTree model produces:

| Metric    |  Result |
| --------- | ------: |
| R-Squared |  0.8862 |
| MAE       | $0.6191 |
| RMSE      | $3.2357 |
| MSE       | 10.4698 |

At first glance, an R-Squared of approximately **0.89** looks reasonably good.

But examining the 20 worst predictions reveals errors such as:

```text
Actual      Predicted       Error
$320.00       $53.95       $266.05
$275.00       $43.98       $231.02
$297.00       $74.24       $222.76
```

Aggregate metrics do not tell the entire story.

---

## Ah-Hah #2 — Look at the Data Before Blaming the Algorithm

Profiling the original dataset reveals:

```text
Total rows:                     1,048,575
Zero distance:                      5,719
Zero time:                          2,385
Zero distance AND zero time:        1,669

Fare > $100:                          339
Fare > $100 with zero distance:       128
```

Fare distribution:

```text
Minimum Fare:         $2.50
Maximum Fare:       $425.00
Median Fare:          $9.00
95th Percentile:     $30.00
99th Percentile:     $52.00
```

Only a tiny percentage of trips have extremely high fares.

Some of those extreme observations also contain missing or questionable trip measurements.

The lesson:

> **Before changing the model, investigate the data.**

---

# Basic vs. Full Features

## Basic Feature Set

The baseline models use:

* Passenger Count
* Trip Time
* Trip Distance
* Payment Type

## Full Feature Set

The full models add:

* Vendor ID
* Rate Code

The complete feature set is therefore:

```text
PassengerCount
TripTime
TripDistance
PaymentType
VendorId
RateCode
```

Categorical values such as `PaymentType`, `VendorId`, and `RateCode` are encoded before being supplied to the regression model.

---

# Data Cleaning

For this exercise, the cleaning rule is intentionally simple.

Rows are removed when:

```text
TripDistance <= 0
OR
TripTime <= 0
OR
FareAmount <= 0
```

The cleaning operation removes:

```text
Original rows: 1,048,575
Clean rows:    1,042,140
Rows removed:      6,435
Removed:            0.61%
```

Only **0.61% of the dataset** is removed.

This makes the resulting change in model performance particularly interesting.

The cleaning rule is deliberately conservative. It is not intended to represent a complete taxi-data validation strategy.

One of the suggested exercises is to develop better anomaly-detection and business-rule cleaning logic.

---

# Final Results

| Run   | Data     | Features | Algorithm  |  R-Squared |        MAE |       RMSE |        MSE |
| ----- | -------- | -------- | ---------- | ---------: | ---------: | ---------: | ---------: |
| **A** | Original | Basic    | FastTree   |     0.8862 |     0.6191 |     3.2357 |    10.4698 |
| **B** | Original | Full     | FastTree   |     0.9479 |     0.4334 |     2.1896 |     4.7943 |
| **C** | Clean    | Basic    | FastTree   |     0.9207 |     0.5164 |     2.6246 |     6.8885 |
| **D** | Clean    | Full     | FastTree   |     0.9678 |     0.3863 |     1.6717 |     2.7945 |
| **E** | Clean    | Full     | LightGBM   | **0.9684** | **0.3690** | **1.6573** | **2.7466** |
| **F** | Clean    | Full     | FastForest |     0.9174 |     1.2113 |     2.6776 |     7.1697 |

---

# What Did We Learn?

## Feature Engineering Had a Large Effect

Compare Run A with Run B:

```text
                         Run A       Run B
R-Squared                0.8862      0.9479
MAE                      0.6191      0.4334
RMSE                     3.2357      2.1896
```

The algorithm did not change.

The data did not change.

Only the available features changed.

Adding `VendorId` and `RateCode` produced a substantial improvement.

### Lesson

> **The model cannot learn from business information it never receives.**

---

## Data Cleaning Also Had a Significant Effect

Compare Run A with Run C:

```text
                         Run A       Run C
R-Squared                0.8862      0.9207
MAE                      0.6191      0.5164
RMSE                     3.2357      2.6246
```

The feature set and algorithm remained unchanged.

Only the data changed.

### Lesson

> **Improving data quality can improve the model without changing the algorithm.**

---

## Better Data + Better Features Work Together

Run D combines clean data with the full feature set:

```text
R-Squared: 0.9678
MAE:       0.3863
RMSE:      1.6717
```

Compare that with the original baseline:

```text
Baseline R-Squared:     0.8862
Improved R-Squared:     0.9678

Baseline MAE:           0.6191
Improved MAE:           0.3863

Baseline RMSE:          3.2357
Improved RMSE:          1.6717
```

The same FastTree algorithm is used in both cases.

### Lesson

> **Data quality and feature engineering can matter more than changing algorithms.**

---

# What About the Algorithm?

Once the data and features are improved, the exercise compares three regression trainers:

* FastTree
* LightGBM
* FastForest

## FastTree

```text
R-Squared: 0.9678
MAE:       0.3863
RMSE:      1.6717
```

## LightGBM

```text
R-Squared: 0.9684
MAE:       0.3690
RMSE:      1.6573
```

## FastForest

```text
R-Squared: 0.9174
MAE:       1.2113
RMSE:      2.6776
```

LightGBM produced the best result, but only slightly improved upon FastTree.

FastForest performed substantially worse for this dataset and configuration.

### Lesson

> **Changing algorithms does not automatically produce a better model.**

More importantly:

> **The improvement from better data and better features was substantially greater than the improvement from switching FastTree to LightGBM.**

---

# The Core Predictive AI Lesson

A common temptation when building predictive systems is to immediately ask:

> Which machine-learning algorithm should I use?

This experiment suggests a better sequence:

```text
Understand the business problem
        ↓
Understand the data
        ↓
Establish a baseline
        ↓
Inspect model failures
        ↓
Improve data quality
        ↓
Improve the features
        ↓
Compare algorithms
        ↓
Tune the winning model
```

In this exercise:

```text
Baseline
R² = 0.8862

        ↓ Better Features

R² = 0.9479

        ↓ Better Data + Features

R² = 0.9678

        ↓ Different Algorithm

R² = 0.9684
```

That progression is one of the most important lessons in practical Predictive AI.

---

# Understanding the Metrics

## R-Squared

R-Squared measures how much of the variance in the target variable is explained by the model.

Generally:

```text
Closer to 1.0 = better
```

The best model in this exercise achieves:

```text
R² = 0.9684
```

---

## MAE — Mean Absolute Error

MAE measures the average absolute difference between the predicted fare and the actual fare.

Because the target is taxi fare, MAE is particularly easy to interpret.

For the LightGBM model:

```text
MAE = $0.3690
```

Across the test population, the model's average absolute error is approximately **37 cents**.

---

## RMSE — Root Mean Squared Error

RMSE penalizes large errors more heavily than MAE.

This makes RMSE particularly useful for detecting models that perform well most of the time but occasionally make very large mistakes.

The large difference between MAE and RMSE in the initial model encouraged us to investigate the worst predictions.

That investigation led directly to the data-quality experiments.

---

## MSE — Mean Squared Error

MSE squares each prediction error before averaging them.

Large prediction errors therefore have a disproportionately large effect on MSE.

RMSE is simply the square root of MSE and is usually easier to interpret because it returns the error to the same units as the target variable.

---

# Worst-Prediction Analysis

Each experiment prints the 20 predictions with the largest absolute errors.

Example:

```text
Actual      Predicted       Error      Distance     Time     Pass   Payment   Rate   Vendor
------------------------------------------------------------------------------------------------
$260.00       $57.13       $202.87       20.10      1360       1      CRD       5      CMT
$332.00      $130.43       $201.57       24.40      4969       1      CRD       4      CMT
$270.00       $77.87       $192.13       20.50      1666       1      CRD       5      CMT
```

This is intentional.

A model should not be evaluated only by a single summary number.

Error analysis can reveal:

* bad source data
* missing variables
* unusual business cases
* rare observations
* model limitations
* different populations hidden inside the dataset

---

# Experiment Questions

When running the project, consider these questions:

### A → B

How much did additional business context and features improve the model?

### A → C

How much did basic data cleaning improve the model?

### C → D

After cleaning the data, how much additional benefit came from the full feature set?

### D → E

Does LightGBM outperform FastTree when given identical data and features?

### D → F

Does FastForest outperform FastTree?

### E → F

How significant is the difference between algorithms compared with the effect of data and features?

---

# Try It Yourself

The existing experiments are intentionally only a starting point.

Try modifying the application.

## Experiment With Cleaning Rules

What happens if you remove trips with:

```text
TripDistance < 0.1 miles
```

Or extremely high distances?

What about records where:

```text
Fare = $2.50
Distance = 55 miles
```

Are they bad data, valid edge cases, or evidence that another variable is missing?

---

## Remove Individual Features

Try removing:

```text
RateCode
```

Then:

```text
VendorId
```

Then:

```text
TripTime
```

Which feature has the largest effect?

---

## Add Feature Engineering

Create derived features such as:

```text
MilesPerMinute
FarePerMile
TripDurationMinutes
```

Be careful not to create features that leak the value you are trying to predict.

---

## Try Other ML.NET Trainers

Add additional regression algorithms and compare them using exactly the same training/test data.

Questions to consider:

* Does the new model improve R-Squared?
* Does it improve MAE?
* Does it improve RMSE?
* Does it reduce the largest prediction errors?
* How much additional training time does it require?

---

## Tune Hyperparameters

The current exercise uses essentially default trainer settings.

Try changing:

* number of trees
* number of leaves
* minimum examples per leaf
* learning rate
* tree depth

Then determine whether tuning produces a meaningful improvement over the baseline configuration.

---

# Requirements

* Visual Studio 2026 or another compatible .NET development environment
* .NET 10 SDK
* C#
* ML.NET 5.0

NuGet packages used by the project:

```text
Microsoft.ML
Microsoft.ML.FastTree
Microsoft.ML.LightGbm
```

---

# Getting Started

Clone the repository:

```bash
git clone https://github.com/AI-n-DotNet/AInDotNet.MLNET.TaxiFare.git
```

Open the solution in Visual Studio.

Restore the NuGet packages if Visual Studio does not restore them automatically.

---

# Dataset

The application expects the training dataset at:

```text
Data/taxi-fare-train.csv
```

The CSV schema used by the exercise is:

```text
VendorId
RateCode
PassengerCount
TripTime
TripDistance
PaymentType
FareAmount
```

The dataset is based on the taxi fare data used in Microsoft's ML.NET taxi fare prediction examples.

If the data file is not present, the application displays:

```text
Training data file not found.
See Data/README.md for download instructions.
```

See the `Data/README.md` file in this repository for dataset setup instructions.

---

# Project Structure

```text
AInDotNet.MLNET.TaxiFare
│
├── Data
│   ├── README.md
│   └── taxi-fare-train.csv
│
├── ExperimentResult.cs
├── Program.cs
├── TaxiFare.cs
├── TaxiFareEvaluation.cs
├── TaxiTrip.cs
│
├── AInDotNet.MLNET.TaxiFare.csproj
└── AInDotNet.MLNET.TaxiFare.slnx
```

### `Program.cs`

Application entry point.

### `TaxiFare.cs`

Contains the experiment runner, model pipelines, data profiling, evaluation, and output logic.

### `TaxiTrip.cs`

Defines the input data schema.

### `TaxiFareEvaluation.cs`

Defines the prediction output schema used during error analysis.

### `ExperimentResult.cs`

Stores metrics from each experiment for the final comparison table.

---

# Reproducibility

The application uses fixed random seeds:

```csharp
new MLContext(seed: 1);
```

and:

```csharp
TrainTestSplit(... seed: 123);
```

Runs A and B share the same original-data split.

Runs C, D, E, and F share the same cleaned-data split.

This is intentional.

Changing the feature set or algorithm while keeping the underlying train/test observations identical provides a more meaningful controlled comparison.

---

# Suggested Learning Sequence

If you are using this repository as a learning exercise, do not simply run the application and look at the final table.

Work through it incrementally:

1. Run the baseline.
2. Examine the model metrics.
3. Examine the worst predictions.
4. Profile the dataset.
5. Form a hypothesis.
6. Add features.
7. Measure the change.
8. Clean the data.
9. Measure the change.
10. Combine data cleaning and feature improvements.
11. Compare algorithms.
12. Investigate the remaining failures.
13. Develop your own experiments.

The important skill is not memorizing ML.NET syntax.

The important skill is learning how to **reason experimentally about predictive models**.

---

# About AInDotNet

[AInDotNet](https://AInDotNet.com) focuses on practical enterprise AI using Microsoft technologies including:

* C#
* .NET
* ML.NET
* Microsoft Azure AI
* Semantic Kernel
* Predictive AI
* Intelligent Document Processing
* AI assistants
* enterprise AI architecture

Visit:

**https://AInDotNet.com**

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

## Final Takeaway

If you remember only one thing from this exercise, remember this:

> **Predictive AI performance is not just about picking the fanciest model.**

In this experiment, understanding the data and supplying better business context produced much larger improvements than simply switching algorithms.

**Data first. Features second. Model tuning third.**
