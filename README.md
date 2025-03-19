# Chronic Kidney Disease (CKD) Prediction Web Application

This repository contains a web application designed to predict the likelihood of Chronic Kidney Disease (CKD) based on patient data. It utilizes a machine learning model trained on the Kaggle CKD dataset and provides a user-friendly interface for making predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project aims to provide a simple and accessible tool for predicting CKD. By inputting relevant patient data, users can obtain a prediction indicating whether the patient is likely to have CKD. The application is built using Flask for the backend and HTML/JavaScript for the frontend, with a machine learning model trained using scikit-learn.

## Features

- **User-friendly Web Interface:** An intuitive interface for inputting patient data.
- **CKD Prediction:** Predicts the likelihood of CKD based on input features.
- **Sample Data Buttons:** Buttons to quickly populate the form with sample data for testing.
- **Clear Output Display:** Displays the prediction result clearly on the page.
- **Error Handling:** Graceful handling of input errors with user-friendly error messages.
- **Clear Output Button:** A button to clear the prediction results.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/Siddharth0207/CKD-Complete-Version.git](https://www.google.com/search?q=https://github.com/Siddharth0207/CKD-Complete-Version.git)
    cd CKD-Complete-Version
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Flask Application:**

    ```bash
    python application.py
    ```

2.  **Open the Web Application:**

    Open your web browser and navigate to `http://0.0.0.0:5000/`.

3.  **Input Patient Data:**

    Enter the required patient data in the input fields.

4.  **Make a Prediction:**

    Click the "Predict your CKD Stage" button to get the prediction.

5.  **Use Sample Data:**

    Click the "Fill CKD Sample Data" or "Fill Not CKD Sample Data" buttons to populate the form with sample data for testing purposes.

6.  **Clear Output:**

    Click the "Clear Output" button to remove the prediction result.

## Project Structure
CKD-Complete-Version/
├── .dvc/                          # Data Version Control files
├── .ebextensions/                 # AWS Elastic Beanstalk configuration
│   └── python.config
├── .github/                       # GitHub Actions workflows (if any)
├── artifacts/                     # Artifacts from training/processing
├── catboost_info/                 # CatBoost model info (if used)
├── data_science_project.egg-info/ # Python package info
├── logs/                          # Application logs
├── notebook/                      # Jupyter notebooks for exploration/training
├── src/                           # Source code directory
├── templates/                     # HTML templates
├── venv/                          # Virtual environment (if used)
├── .dvcignore                     # DVC ignore file
├── .gitignore                     # Git ignore file
├── application.py                 # Flask application file
├── ckd.py                         # Main application logic
├── Dockerfile                     # Docker configuration
├── README.md                      # Project documentation
├── requirements.txt               # Project dependencies
├── setup.py                       # Setup script for packaging
├── template.py                    # Template file (if used)

## Model Details

The machine learning model used in this application is trained on the Kaggle CKD dataset. It utilizes a KNN or Random Forest and is trained to predict the likelihood of CKD based on patient features. The model is saved as  `model.pkl` in the `artifacts/` directory.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Push your changes to your fork.
5.  Submit a pull request.

## Acknowledgements

-   The CKD dataset. [https://www.kaggle.com/datasets/mansoordaku/ckdisease]
-   The CatBoost library for efficient and scalable machine learning. [https://catboost.ai/]
-   The Flask web framework for building web applications. [https://flask.palletsprojects]
-   CookieCutter for project templating. [https://github.com/cookiecutter/cookiecutter-pyp]
-   DVC for data version control. [https://dvc.org/]
-   DagsHub for data science project management. [https://dags.ai/]
-   mlflow for model tracking and deployment. [https://mlflow.org/]
-   Azure Machine Learning for model deployment. [https://azure.microsoft.com/en-us/services/machine-learning/]
-   GitHub Actions for continuous integration and deployment. [https://github.com/features/actions]
---
