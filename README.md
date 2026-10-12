# Adult Income Prediction & Fairness Audit

This project uses machine learning to predict whether a person earns more than $50K a year, using the UCI Adult Income dataset (taken from the 1994 US Census database). It started as a group project for my data science course. Afterwards I went back to the same data to compare the models more fairly and to check whether they treat men and women, and White and non-White people, differently.

## What's in this repo

- `group-project.ipynb` is the original notebook from our course project, kept as we submitted it.
- `adult-income-fairness-audit.ipynb` is my follow-up work. Start here.
- `Project Data (Adult).csv` is the dataset. Both notebooks read it, so keep this exact file name and keep it in the same folder.
- `images/` has the charts used in this README.

## The group project

We cleaned the data, picked features, encoded categories, scaled numbers, and trained Logistic Regression, Random Forest and XGBoost. We compared accuracy, precision, recall, F1-score and confusion matrices.

**My contribution:** feature selection and encoding. I tested different combinations of columns and looked at how each one changed model performance.
*(Edit this so it says exactly what you did.)*

One thing we found: the best columns depended on the model. For logistic regression, the best set in our tests left out age, fnlwgt, marital status and race.

## My follow-up

After the course I wanted to look past overall accuracy, because a model can score well and still make different kinds of mistakes for different groups. The follow-up uses only pandas, numpy, scikit-learn and matplotlib. It uses scikit-learn's `HistGradientBoostingClassifier` instead of XGBoost, since it works in a similar way.

### 1. A fairer model comparison

Every model uses the same stratified 80/20 split. I used 5-fold cross-validation (ROC-AUC) on the training data to compare models, and used the test set once at the end. The group project compared models on different splits, so its numbers weren't directly comparable.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.843 | 0.656 | 0.903 |
| Random Forest | 0.847 | 0.674 | 0.901 |
| Gradient Boosting | 0.862 | 0.697 | 0.922 |

Gradient boosting is the best of the three. All three models still miss about 36 to 40% of the people who really earn more than $50K.

<img width="544" height="494" alt="roc-curves" src="https://github.com/user-attachments/assets/a283a0bf-325f-446e-9259-cfa1adda7f46" />

### 2. A fairness audit

The models predict "earns more than $50K" for about 26% of men but only about 9% of women. That is a ratio of about 0.32 (a common rule of thumb says anything below 0.8 deserves a closer look). Part of this comes from the data itself: about 31% of men and 11% of women in the data earn more than $50K.

I checked:
- how often each group is predicted to earn more than $50K, and the ratio between groups
- recall among people who really earn more than $50K
- true positive and false positive rates
- bootstrap 95% intervals, to see how uncertain the gaps are

### 3. Two attempts to reduce the gap

| Approach | Sex gap (logistic regression) | Sex gap (gradient boosting) | Accuracy change |
|---|---|---|---|
| Baseline | -0.179 | -0.177 | |
| Delete the `sex` and `race` columns | -0.165 | -0.174 | almost none |
| Reweigh the training data on sex | -0.079 | -0.088 | about -0.6 to -0.8 points |

A gap of 0 would mean both groups get "above $50K" predictions at the same rate.

<img width="651" height="234" alt="fairness-gaps" src="https://github.com/user-attachments/assets/6c931466-0250-4a97-b66d-f24451728e1a" />

- **Deleting sensitive columns barely helped.** A model can guess someone's sex from the other columns with ROC-AUC 0.94, so the information is still there.
- **Reweighing roughly halved the sex gap** and held up over five random splits. I wrote the method myself (Kamiran and Calders' reweighing, also in IBM's AI Fairness 360 toolkit). The cost was a small drop in accuracy and F1.
- **Fixing one measure moved another.** After reweighing, women who really earn more than $50K were flagged about 20 points more often than men, so the recall gap flipped direction.
- **Reweighing on race** reduced the race gap with almost no loss in accuracy, but I only tested it on one split, so I treat it as a first look.

## What I learned

*(Write this in your own words. A few ideas: what you noticed about removing sensitive columns, why you can't just look at accuracy, and what you'd want to know before using a model like this.)*

## How to run it

1. Install Python 3 and the libraries:
   ```bash
   python3 -m pip install pandas numpy matplotlib scikit-learn jupyter
   ```
2. Put `adult-income-fairness-audit.ipynb` next to `Project Data (Adult).csv`.
3. Start Jupyter and run the cells from top to bottom:
   ```bash
   jupyter notebook
   ```
4. Optional: `python3 -m pip install aif360` lets you run the last check, which compares my hand-written reweighing against IBM's version.

The saved results come from Python 3.13, pandas 3.0 and scikit-learn 1.9. Other versions can change the numbers slightly.

## Limitations

- The data is from 1994 and doesn't describe today's population.
- Sex is only male or female in this data, and I combined all non-White groups into one, which hides differences between them. I didn't look at combined groups such as Black women.
- In the group project we compared feature sets using the test set, so small differences between them may just be noise.
- Different fairness measures disagree with each other, as the reweighing result shows.
- All models use one fixed cutoff (0.5). Other cutoffs would change the numbers.
- The audit measures differences in predictions. It doesn't explain why they exist, and it isn't proof that a model is fair.

## Credits

- Dataset: Becker, B. and Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. Check the dataset page for the exact citation and licence.
- Reweighing: Kamiran, F. and Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33, 1-33. Also implemented in IBM's [AI Fairness 360](https://github.com/Trusted-AI/AIF360).
- I used Claude as a tutor and coding assistant for the follow-up, and I ran the code and checked the results myself. *(Edit this line so it matches how you actually worked.)*
