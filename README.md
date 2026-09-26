# Adult Income Prediction

This project uses machine learning to predict whether a person's annual income is above or below $50K based on different demographic and employment-related factors.

## About the Project

For this project, we worked with the Adult Income dataset and went through the process of preparing the data, training different machine learning models, and comparing their performance.

The dataset includes information such as age, education, occupation, marital status, hours worked per week, work class, and other demographic information.

## What We Did

Before training the models, we cleaned and prepared the dataset by handling unknown values, selecting features, encoding categorical data, and scaling numerical data.

We then tested different machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

To compare the models, we looked at accuracy, precision, recall, F1-score, and confusion matrices.

## What We Found

One of the main things we explored was how different features affected the model's predictions. We tested different combinations of features and found that feature selection could have a noticeable impact on model performance.

One interesting result was that removing marital status gave us the highest accuracy out of the feature combinations we tested.

## My Contribution

This was a group project, and my main responsibility was working on feature selection and encoding. I helped prepare the data for the models, tested different combinations of features, and looked at how those changes affected the results.

## Tools & Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## What I Learned

This project gave me experience working through a full machine learning project instead of only focusing on building a model. I learned how important data cleaning, feature selection, and preprocessing are and how they can directly affect the final results.

It also gave me more experience comparing different models and using metrics beyond just accuracy to understand how well a model is actually performing.

## Files

- `adult-income-analysis.ipynb` — Jupyter Notebook containing the analysis, preprocessing, models, and results
- `adult-income-data.csv` — Dataset used for the project
