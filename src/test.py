import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

heart_data = pd.read_csv('../data/cardio_train.csv', delimiter=';')

heart_data.head()

heart_data.info()

#Output:
#  <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 70000 entries, 0 to 69999
# Data columns (total 13 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   id           70000 non-null  int64  
#  1   age          70000 non-null  int64  
#  2   gender       70000 non-null  int64  
#  3   height       70000 non-null  int64  
#  4   weight       70000 non-null  float64
#  5   ap_hi        70000 non-null  int64  
#  6   ap_lo        70000 non-null  int64  
#  7   cholesterol  70000 non-null  int64  
#  8   gluc         70000 non-null  int64  
#  9   smoke        70000 non-null  int64  
#  10  alco         70000 non-null  int64  
#  11  active       70000 non-null  int64  
#  12  cardio       70000 non-null  int64  
# dtypes: float64(1), int64(12)
# memory usage: 6.9 MB

heart_data['age'] = (heart_data['age'] / 365).round().astype('int64')
heart_data['bmi'] = heart_data['weight'] / ((heart_data['height'] / 100) ** 2)

heart_data = heart_data.drop(heart_data[heart_data['ap_hi'] <= heart_data['ap_lo']].index)
heart_data = heart_data.drop(heart_data[heart_data['ap_hi'] <= 0].index)
heart_data = heart_data.drop(heart_data[heart_data['ap_lo'] <= 0].index)

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Giữ lại các giá trị trong khoảng [lower_bound, upper_bound]
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

columns_to_remove = ['age','ap_hi','ap_lo','bmi']
for column in columns_to_remove:
    heart_data = remove_outliers(heart_data,column)


X = heart_data.drop(columns=['cardio','id'])
y = heart_data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


num_features = ['age', 'bmi', 'ap_hi', 'ap_lo']
cat_features = ['cholesterol','gluc','gender']

num_pipeline = Pipeline([
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder()),
])

preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

X_train = preprocess_pipeline.fit_transform(X_train)
X_test = preprocess_pipeline.fit_transform(X_test)

dt_model = DecisionTreeClassifier(criterion='log_loss', max_depth=6, max_features=7, splitter='best')
random_forest_model = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=300, max_features='sqrt')
svc_model = SVC(C=1, gamma=0.1, kernel='rbf', probability=True)  # Thêm probability=True để SVC có thể dự đoán xác suất
ada_boost_model = AdaBoostClassifier(n_estimators=300, learning_rate=0.7,algorithm='SAMME.R')
gradient_boosting_model = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=300)

voting_classifier = VotingClassifier(estimators=[
    ('dt', dt_model),
    ('rf', random_forest_model),
    ('svc', svc_model),
    ('ada', ada_boost_model),
    ('gb', gradient_boosting_model)
], voting='soft')  # Sử dụng voting='soft' để kết hợp dự đoán dựa trên xác suất

voting_classifier.fit(X_train, y_train)

y_pred = voting_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_pred, y_test, average='weighted')

cv = cross_val_score(voting_classifier, X_train, y_train, cv=5)

report = classification_report(y_pred, y_test, zero_division=1)

print(f"Độ chính xác của Voting Classifier model: {round(accuracy*100, 2)}%\n")
print(f"f1 score của Voting Classifier model: {round(f1score*100, 2)}%\n")
print(f"Cross validation của Voting Classifier model: {round(np.mean(cv)*100, 2)}%\n")
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=voting_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Voting Classifier Confusion Matrix")
plt.grid(visible=False)
plt.show()


import pickle

# Lưu mô hình vào file
filename = 'voting_classifier_model.pkl'
pickle_out = open(filename, 'wb')
pickle.dump(voting_classifier, open(filename, 'wb'))
pickle_out.close()