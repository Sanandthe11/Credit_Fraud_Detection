import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

#Load the data
df =pd.read_csv(r'C:\Users\SHEEJA\Downloads\creditcard.csv\creditcard.csv')
#print(df.head(7))

#Drop columns and standardising the amount
df = df.drop("Time", axis=1)
scaler = preprocessing.StandardScaler()
df['std_Amount'] = scaler.fit_transform(df['Amount'].values.reshape (-1,1))
df = df.drop("Amount", axis=1)

#plot the data to find the distribution of type of transactions
sns.countplot(x='Class',data=df)
plt.show()

# Create depented and independed variables
cols = df.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"
X = df[cols]
Y = df[target]

# Create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# XGboost Setup
xgb = XGBClassifier(random_state=2, use_label_encoder=False, eval_metric='logloss')
param_dist = {'scale_pos_weight': stats.randint(1, 50),
    'max_depth': stats.randint(3, 10),
    'min_child_weight': stats.randint(1, 20),
    'subsample': stats.uniform(0.6, 0.4),
    'colsample_bytree': stats.uniform(0.6, 0.4),
    'learning_rate': stats.uniform(0.01, 0.2),
    'n_estimators': stats.randint(50, 200)
}

# Use stratified k-fold to maintain class distribution in splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with F1 score as scoring metric (good balance of precision/recall)
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=50, scoring='f1', cv=cv, verbose=2, n_jobs=-1, random_state=42)

# Fit grid search on training data
random_search.fit(X_train, y_train)

# Best parameters from tuning
print("Best parameters:", random_search.best_params_)

# Best model after tuning
best_xgb = random_search.best_estimator_

# Predictions on test set
y_pred_tuned = best_xgb.predict(X_test)

print("Accuracy XGB:",metrics.accuracy_score(y_test, y_pred_tuned))
print("Precision XGB:",metrics.precision_score(y_test, y_pred_tuned))
print("Recall XGB:",metrics.recall_score(y_test, y_pred_tuned))
print("F1 Score XGB:",metrics.f1_score(y_test, y_pred_tuned))

matrix_xgb = confusion_matrix(y_test, y_pred_tuned)
cm_xgb = pd.DataFrame(matrix_xgb, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_xgb, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix XGBoost"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

y_pred_xgb_proba = best_xgb.predict_proba(X_test)[::,1]
fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y_test,  y_pred_xgb_proba)
auc_xgb = metrics.roc_auc_score(y_test, y_pred_xgb_proba)
print("AUC XGBoost :", auc_xgb)

plt.plot(fpr_xgb,tpr_xgb,label="XGBoost, auc={:.3f})".format(auc_xgb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost ROC curve')
plt.legend(loc=4)
plt.show()

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_pred_xgb_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(xgb_recall, xgb_precision, color='orange', label='XGB')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
