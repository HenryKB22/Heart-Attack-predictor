# Heart-Attack-predictor

Heart Attack Prediction Model
Overview
This project develops a machine learning model to predict the likelihood of a heart attack based on patient health data. The dataset includes features such as age, gender, heart rate, blood pressure, blood sugar, and cardiac biomarkers (CK-MB and Troponin). Two models, Logistic Regression and XGBoost, are trained and evaluated to classify outcomes as either "positive" (heart attack) or "negative" (no heart attack).
Dataset
The dataset (heart_attack_data.csv) contains 1,319 patient records with the following features:

Age: Patient's age (integer).
Gender: Patient's gender (originally encoded as 1 for Male, 0 for Female; transformed to "Male"/"Female").
Heart rate: Patient's heart rate in beats per minute (integer).
Systolic blood pressure: Systolic blood pressure in mmHg (integer).
Diastolic blood pressure: Diastolic blood pressure in mmHg (integer).
Blood sugar: Blood sugar level in mg/dL (float).
CK-MB: Creatine Kinase-MB level, a cardiac biomarker (float).
Troponin: Troponin level, another cardiac biomarker (float).
Result: Target variable indicating heart attack outcome ("positive" or "negative").

Dataset Summary

Total Records: 1,319
Gender Distribution: 870 Males, 449 Females
Outcome Distribution: 810 Positive (heart attack), 509 Negative (no heart attack)
Numerical Features: Age, Heart rate, Systolic blood pressure, Diastolic blood pressure, Blood sugar, CK-MB, Troponin
Categorical Features: Gender (after transformation)

Project Structure

Heart_attack.ipynb: Jupyter Notebook containing the complete code for data loading, preprocessing, visualization, model training, and evaluation.
heart_attack_data.csv: Input dataset used for training and testing.
cleaned_heart_attack_data.csv: Processed dataset with transformed gender values (downloaded from the notebook).

Dependencies
To run the project, install the required Python packages:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Data Preprocessing
The preprocessing steps are implemented in the preprocess_for_ridge function, although it is adapted for classification:

Gender Transformation: Converted numeric gender values (1/0) to "Male"/"Female".
Feature Encoding:
Categorical features (e.g., Gender) are one-hot encoded using OneHotEncoder with the drop='first' option to avoid multicollinearity.
Numerical features are standardized using StandardScaler.


Data Splitting: The dataset is split into training (80%) and testing (20%) sets with a random state of 42 for reproducibility.
Handling Missing Columns: The notebook assumes additional columns (Age Group, Blood_sugar_category) that are not present in the provided dataset, which may cause errors unless these are created beforehand.

Note: The preprocessing function references Age Group and Blood_sugar_category, which are not in the dataset. To resolve this, either remove these from the categorical columns list or create these features (e.g., by binning Age and Blood sugar).
Model Training
Two machine learning models are trained to predict heart attack outcomes:

Logistic Regression:

Algorithm: Logistic Regression with a maximum of 1,000 iterations.
Parameters: random_state=42 for reproducibility.
Performance:
Accuracy: 79.17%
Classification Report:precision    recall  f1-score   support
negative       0.76      0.67      0.71       101
positive       0.81      0.87      0.84       163
accuracy                           0.79       264
macro avg      0.78      0.77      0.77       264
weighted avg   0.79      0.79      0.79       264






XGBoost:

Algorithm: XGBoost Classifier with label encoding for the target variable.
Parameters: use_label_encoder=False, eval_metric='logloss', random_state=42.
Performance:
Accuracy: 98.11%
Classification Report:precision    recall  f1-score   support
negative       0.98      0.97      0.98       101
positive       0.98      0.99      0.98       163
accuracy                           0.98       264
macro avg      0.98      0.98      0.98       264
weighted avg   0.98      0.98      0.98       264







Data Visualization
A heatmap of correlations between numerical features is generated using Seaborn to explore relationships between variables like Age, Heart rate, and Troponin.
How to Run

Clone the Repository:git clone <repository-url>
cd <repository-directory>


Install Dependencies:pip install -r requirements.txt

(Create a requirements.txt file with the listed dependencies if needed.)
Run the Notebook:
Open Heart_attack.ipynb in Jupyter Notebook or JupyterLab.
Ensure heart_attack_data.csv is in the same directory.
Execute the cells sequentially.


Fix Preprocessing Error (if applicable):
Modify the preprocess_for_ridge call to include only available categorical columns:X_train, X_test, y_train, y_test, preprocessor = preprocess_for_ridge(df, 'Result', ['Gender'])


Alternatively, create Age Group and Blood_sugar_category features before preprocessing, e.g.:df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])
df['Blood_sugar_category'] = pd.cut(df['Blood sugar'], bins=[0, 100, 200, 300, 600], labels=['Low', 'Normal', 'High', 'Very High'])





Results

Logistic Regression: Achieves moderate performance with an accuracy of 79.17%. It performs better for positive cases (heart attack) with higher recall (0.87) but struggles with negative cases (0.67 recall).
XGBoost: Significantly outperforms Logistic Regression with a 98.11% accuracy, showing high precision and recall for both classes.
Conclusion: XGBoost is the superior model for this dataset, likely due to its ability to capture complex patterns and interactions between features.

Future Improvements

Feature Engineering: Create Age Group and Blood_sugar_category features to leverage additional categorical information.
Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to optimize Logistic Regression and XGBoost parameters.
Cross-Validation: Implement k-fold cross-validation to ensure robust model evaluation.
Outlier Handling: Address potential outliers in features like Heart rate (max: 1111 bpm) and CK-MB (max: 300).
Feature Importance: Analyze feature importance from XGBoost to understand key predictors of heart attacks.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or contributions, please open an issue or submit a pull request on this repository.
