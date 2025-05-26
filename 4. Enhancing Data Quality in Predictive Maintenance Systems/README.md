# Enhancing Data Quality in Predictive Maintenance Systems

This project presents a structured approach to improve the quality of sensor data used in predictive maintenance systems. It includes steps like missing value handling, outlier detection, normalization, and visualization. Cleaned data improves the performance and reliability of downstream machine learning models for failure prediction.

## Project Phases

### 1. Research Phase
- Identified common data quality issues in predictive maintenance:
  - Missing values
  - Duplicated entries
  - Sensor outliers
  - Inconsistent scales
- Reviewed predictive maintenance case studies and datasets (e.g., NASA CMAPSS, Kaggle sensor datasets).
- Selected a sample dataset that simulates real-world machine sensor readings.

### 2. Design Phase
- Designed a modular data preprocessing pipeline with the following components:
  - **Missing value handler**: Rows with null values are removed.
  - **Duplicate remover**: Duplicate entries are identified and dropped.
  - **Outlier detector**: Visualized using boxplots to understand data range and anomalies.
  - **Normalization unit**: Sensor data standardized using Z-score normalization.
- Created a directory structure to organize raw data, scripts, and results.

### 3. Development Phase
- Implemented Python scripts for preprocessing using:
  - `pandas` for data handling
  - `seaborn` and `matplotlib` for outlier visualization
  - `scikit-learn` for scaling
- Saved cleaned outputs for further use in predictive models.

### 4. Testing Phase
- Verified preprocessing with test cases:
  - Confirmed removal of all missing/duplicate records.
  - Validated that outlier ranges were visualized in generated plots.
  - Normalized data was within expected ranges (mean ≈ 0, std ≈ 1).
- Checked robustness by applying the pipeline to slightly modified versions of the dataset.

### 5. Deployment Phase (GitHub)
- Uploaded the project to GitHub with clear folder structure:
  - `data/` — raw input dataset
  - `scripts/` — data cleaning and preprocessing logic
  - `results/` — cleaned data and charts
  - `README.md` — this documentation
- Included `requirements.txt` to ensure reproducibility
- Prepared the project for public access and future integration into ML pipelines

### File Structure

```
.
├── train_FD001.txt                  # NASA CMAPSS training data
├── predictive_maintenance.csv      # Validation/test dataset
├── predictive_maintenance_lstm_model.h5  # Trained LSTM model
├── app.py                          # Flask app script
└── README.md                       # Project documentation
```


