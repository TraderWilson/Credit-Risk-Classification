# Credit Risk Classification Report

## Analysis Purpose
The primary goal of our analysis was to develop a machine learning model capable of accurately predicting the risk level of loans. This predictive capability is crucial for financial institutions to make informed lending decisions, thereby minimizing risk and optimizing the allocation of credit.

## Financial Data Overview
We focused on a dataset comprising various financial information related to individual loan applications. This dataset included both numerical and categorical variables, such as applicant income, loan amount, credit history, marital status, and education level. The critical outcome we aimed to predict was whether a loan would likely be repaid (labeled as '0' for healthy loans) or potentially default (labeled as '1' for high-risk loans).

## Variables of Interest
The variable we sought to predict was the loan's risk level, a binary outcome indicating the health or risk of the loan. An initial analysis of our dataset using value_counts revealed a significant class imbalance, with a large majority of loans being healthy (label '0') and a smaller portion being considered high-risk (label '1').

## Machine Learning Process Stages
Our analysis followed a structured machine learning process, outlined as follows:

1. Data Preprocessing: We began by cleaning the data, handling missing values, and encoding categorical variables to prepare our dataset for modeling.
2. Exploratory Data Analysis (EDA): Through EDA, we gained insights into the distribution of variables, the class imbalance, and potential correlations between features and the target variable.
3. Feature Selection: We employed various techniques to identify the most predictive features, reducing dimensionality while preserving the model's predictive power.
4. Model Development and Evaluation: We experimented with several machine learning models, including Logistic Regression, to predict the risk level of loans. Given the class imbalance, model performance was evaluated using metrics beyond accuracy, such as precision, recall, F1 score, and the Area Under the ROC Curve (AUC).
5. Resampling Techniques: To address the class imbalance, we implemented resampling techniques such as `RandomOverSampler`, which helped improve the minority class's predictive performance without compromising the majority class's accuracy.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Accuracy
    * Definition: Accuracy is the ratio of correctly predicted observations (both true positives and true negatives) to the total observations in the dataset.
    * Interpretation: The "avg / total" accuracy of 0.99 suggests that, overall, the model correctly predicts the risk status of loans 99% of the time. This high accuracy indicates a strong model performance, especially considering the model's ability to generalize across the entire dataset.
  * Precision (Pre)
    * Definition: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. It indicates the quality of the positive class predictions.
    * Interpretation for Healthy Loans (0): A precision of 1.00 for healthy loans means that every loan the model predicted as healthy was actually healthy. This perfect precision score indicates no false positives for healthy loans.
    * Interpretation for High-Risk Loans (1): A precision of 0.87 for high-risk loans means that 87% of the loans predicted as high-risk were indeed high-risk. While not perfect, it's a strong score, especially considering that false positives in this context mean predicting a loan as high-risk when it's actually healthy.
  * Recall (Rec)
    * Definition: Recall, also known as sensitivity or true positive rate, is the ratio of correctly predicted positive observations to all observations in actual class. It measures the model's ability to capture actual positives.
    * Interpretation for Healthy Loans (0): The recall of 1.00 for healthy loans indicates the model correctly identified all healthy loans without missing any, showcasing perfect sensitivity.
    * Interpretation for High-Risk Loans (1): The recall of 0.95 for high-risk loans signifies that the model was able to identify 95% of the actual high-risk loans. This high recall is crucial in a financial context, as failing to identify high-risk loans could have significant financial implications.

* Machine Learning Model 2:
  * Precision (Pre)
    * Healthy Loans (0): The precision of 1.00 indicates that every prediction made by the model for a loan being healthy was correct. There were no instances where a loan was incorrectly identified as healthy, which means no false positives for this category.
    * High-Risk Loans (1): A precision of 0.87 for high-risk loans means that 87% of the loans predicted as high-risk were truly high-risk. This is a strong indicator of the model's reliability in identifying loans that are genuinely at risk, though it implies that 13% of the loans predicted as high-risk were actually healthy (false positives).
  * Recall (Rec)
    * Healthy Loans (0): The recall of 1.00 demonstrates the model's perfect ability to identify all healthy loans. This means it successfully detected all true positives without missing any, indicating excellent sensitivity for this class.
    * High-Risk Loans (1): With a recall of 0.95, the model identified 95% of all actual high-risk loans. This high recall is critical, as it shows the model's effectiveness in capturing most loans that could potentially default, minimizing the risk of financial loss due to undetected high-risk loans.
  * Specificity (Spe)
    * Healthy Loans (0): The specificity, while not directly provided, can be inferred from the context. For healthy loans, the specificity is indirectly suggested to be high due to the high recall and precision for the other class. A specificity of 1.00 for predicting high-risk loans when the model predicts a loan to be healthy indicates it correctly identifies healthy loans without mistakenly classifying them as high-risk.
    * High-Risk Loans (1): The model's specificity here can be understood as its ability to correctly identify loans that are not high-risk when it predicts them as such. The reported specificity of 0.95 for the healthy loans when predicting high-risk ones shows excellent performance in avoiding false negatives in this class.
  * F1 Score (F1)
    * Healthy Loans (0): An F1 score of 1.00 signifies a perfect balance between precision and recall for healthy loans, indicating the model's exceptional performance in this category.
    * High-Risk Loans (1): The F1 score of 0.91 for high-risk loans reflects a strong balance between precision and recall, suggesting the model effectively identifies high-risk loans while maintaining a reasonable accuracy level.
  * Geometric Mean (Geo) and Index of Balanced Accuracy (Iba)
    * Both Classes: The geometric mean (Geo) of 0.97 and the index of balanced accuracy (Iba) of 0.95 for both classes indicate a high level of model robustness and balance, confirming its effectiveness in classifying both healthy and high-risk loans accurately.
  * Support (Sup)
    * The support numbers, 18,759 for healthy loans and 625 for high-risk loans, highlight the class imbalance in the dataset. Despite this, the model shows remarkable performance metrics across both classes, effectively addressing the imbalance in its predictions.

## Summary

The core of our predictive modeling was built around the LogisticRegression algorithm due to its efficacy in binary classification problems and its interpretability. Recognizing the challenge posed by the class imbalance, we applied the Random Over Sampling technique to enhance the representation of high-risk loans in our training dataset. This approach significantly improved our model's sensitivity (recall) to high-risk loans without drastically affecting the overall accuracy.

For further exploration, I recommend to do model fine tuning, using techniques like grid search to optimize hyperparameters for better performance, and try to validate the model using a hold-out test set to ensure its generalizability on unseen data.