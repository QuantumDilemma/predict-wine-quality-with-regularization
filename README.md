# Wine Quality Classification using Logistic Regression

This project is about classifying wine quality based on its chemical properties using Logistic Regression.

## Libraries
The following libraries are used:
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

## Data Loading and Preprocessing
The dataset used in this project is the `winequality-red.csv` file. The target variable 'quality' is replaced with a binary class based on the quality score (>5 or not). The data is then split into features and targets, and standardized using the StandardScaler from sklearn.

## Model without Regularization
A Logistic Regression model is built without any regularization. The model's coefficients are plotted and the model is evaluated using the F1 score.

## Model with L2 Regularization
A Logistic Regression model with L2 regularization is built. The F1 score remains the same for the regularized and non-regularized model, indicating that the constraint boundary for regularization is large enough to hold the original loss function minimum.

## Hyperparameter Tuning
The inverse regularization strength 'C' is tuned using GridSearchCV. The F1 scores for different values of 'C' are plotted for both the training and test sets. The model is then evaluated on the test set using the best 'C' value obtained from GridSearchCV.

## Cross-Validation with L1 Regularization
A Logistic Regression model with L1 regularization is built using LogisticRegressionCV. The model is fitted on the entire dataset. The model's coefficients are then plotted. We can see that one feature's coefficient was reduced completely to zero, showing that this method can be used as a feature selection method as well.

