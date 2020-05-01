import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, roc_auc_score


data = pd.read_csv('C:/Users/sucha/Desktop/678/project/BIA 678 dataset.csv')

# print(data.head())

X = data.iloc[:, 1: -1]
y = data.iloc[:, -1]

# print(X.head())
# print(y.head())

X_scale = preprocessing.scale(X)
# print(len(X_scale))
# print(X_scale.shape)
# print(X_scale)

column_name = ['duration', 'goal_usd', 'blurb_length', 'name_length', 'start_month', 'end_month', 'Category', 'Country', 'StartQ', 'EndQ']
X = pd.DataFrame(X_scale, columns=column_name)
# print(X)
X['Status'] = y
# print(X)


data = X
# EDA
df_class_1 = data[data['Status'] == 0]
df_class_0 = data[data['Status'] == 1]

# 0 for 75241
# 1 for 117307

count_class_0, count_class_1 = data.Status.value_counts()

df = pd.concat([df_class_0, df_class_1], axis=0)
print(df.Status.value_counts())
df.Status.value_counts().plot(kind='bar')
plt.show()

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
print('\nFollowing is for oversampling\n')
# print(df_class_1_over)
#
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
print(df_test_over.Status.value_counts())
df_test_over.Status.value_counts().plot(kind='bar')
plt.show()

data = df_test_over
print(data.shape)

X_scale = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X_scale)
print(y)
print(X_scale.shape)
print(y.shape)

pca = PCA(n_components=10)
pca.fit(X_scale)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(var, decimals=4)*100)
# print(var1)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.plot(var1)
plt.show()

A = np.asmatrix(X_scale.T) * np.asmatrix(X_scale)
U, S, V = np.linalg.svd(A)
eigvals = S ** 2 / np.sum(S ** 2)

# print(eigvals)
# print(U)
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(10) +1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)

plt.title('Eigenvalues for Depression Data')
plt.xlabel('Principal Component Number')
plt.ylabel('Eigenvalue')
plt.show()

pca1 = PCA(n_components=7)
X1 = pca1.fit_transform(X_scale)
# print(X1.shape)




X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=0)

LDA = LinearDiscriminantAnalysis()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
NB = GaussianNB()
RF = RandomForestClassifier()
DT = DecisionTreeClassifier()
GBT = GradientBoostingClassifier()



#RF
print('\nRF Classifier')
RF = RandomForestClassifier()
RF = RF.fit(X_train, y_train)
RF_score = RF.score(X_test, y_test)
# print(LR_score)
RF_probs = RF.predict_proba(X_test)
RF_probs = RF_probs[:, 1]
RF_auc = roc_auc_score(y_test, RF_probs)
RF_auc = round(RF_auc, 4)
print(RF_auc)
RF_fpr, RF_tpr, thresholds = roc_curve(y_test, RF_probs, pos_label=1)


# LDA
print('\nLDA Classifier')
LDA = LinearDiscriminantAnalysis()
LDA = LDA.fit(X_train, y_train)
LDA_score = LDA.score(X_test, y_test)
# print(LDA_score)
LDA_probs = LDA.predict_proba(X_test)
LDA_probs = LDA_probs[:, 1]
LDA_auc = roc_auc_score(y_test, LDA_probs)
LDA_auc = round(LDA_auc, 4)
print(LDA_auc)

LDA_fpr, LDA_tpr, thresholds = roc_curve(y_test, LDA_probs, pos_label=1)

# LR
print('\nLR Classifier')
LR = LogisticRegression(solver='lbfgs', max_iter=5000)
LR = LR.fit(X_train, y_train)
LR_score = LR.score(X_test, y_test)
# print(LR_score)
LR_probs = LR.predict_proba(X_test)
LR_probs = LR_probs[:, 1]
# print(probs)
LR_auc = roc_auc_score(y_test, LR_probs)
print(LR_auc)
LR_auc = round(LR_auc, 4)
print(LR_auc)
LR_fpr, LR_tpr, thresholds = roc_curve(y_test, LR_probs, pos_label=1)

#DecisionTree
print('\nDT Classifier')
DT = DecisionTreeClassifier()
DT = DT.fit(X_train, y_train)
DT_score = DT.score(X_test, y_test)
DT_probs = DT.predict_proba(X_test)
DT_probs = DT_probs[:, 1]
DT_auc = roc_auc_score(y_test, RF_probs)
DT_auc = round(LR_auc, 4)
print(DT_auc)
DT_fpr, DT_tpr, thresholds = roc_curve(y_test, DT_probs, pos_label=1)

#gradientboost
print('\nGBT Classifier')
GBT = RandomForestClassifier()
GBT = GBT.fit(X_train, y_train)
GBT_score = GBT.score(X_test, y_test)
GBT_probs = GBT.predict_proba(X_test)
GBT_probs = GBT_probs[:, 1]
GBT_auc = roc_auc_score(y_test, GBT_probs)
GBT_auc = round(GBT_auc, 4)
print(GBT_auc)
GBT_fpr, GBT_tpr, thresholds = roc_curve(y_test, GBT_probs, pos_label=1)

# KNN
print('\nKNeighborsClassifier')
KNN = KNeighborsClassifier(n_neighbors=3)
KNN = KNN.fit(X_train, np.ravel(y_train))
KNN_probs = KNN.predict_proba(X_test)
KNN_probs = KNN_probs[:, 1]
KNN_auc = roc_auc_score(y_test, KNN_probs)
KNN_auc = round(KNN_auc, 4)
print(KNN_auc)

KNN_fpr, KNN_tpr, thresholds = roc_curve(y_test, KNN_probs, pos_label=1)

# NB
print('\nNB Classifier')
NB = GaussianNB()
NB = NB.fit(X_train, y_train)
NB_score = NB.score(X_test, y_test)
# print(NB_score)
NB_probs = NB.predict_proba(X_test)
NB_probs = NB_probs[:, 1]
# print(NB_probs)
NB_auc = roc_auc_score(y_test, NB_probs)
NB_auc = round(NB_auc, 4)
print(NB_auc)

NB_fpr, NB_tpr, thresholds = roc_curve(y_test, NB_probs, pos_label=1)
#
# models plot
plt.plot(RF_fpr, RF_tpr, color='black', label='RF_AUC = {}'.format(RF_auc))
plt.plot(DT_fpr, DT_tpr, color='blue', label='DT_AUC = {}'.format(DT_auc))
plt.plot(LR_fpr, LR_tpr, color='red', label='LR_AUC = {}'.format(LR_auc))
plt.plot(GBT_fpr, GBT_tpr, color='green', label='GBT_AUC = {}'.format(GBT_auc))
#plt.plot(LDA_fpr, LDA_tpr, color='orange', label='LDA_AUC = {}'.format(LDA_auc))
#plt.plot(KNN_fpr, KNN_tpr, color='green', label='KNN_AUC = {}'.format(KNN_auc))
#plt.plot(NB_fpr, NB_tpr, color='blue', label='NB_AUC = {}'.format(NB_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Ensemble
print('\n\n')
estimator = [('NB', NB), ('LR', LR), ('LDA', LDA), ('KNN', KNN)]
ensemble = VotingClassifier(estimator, voting='soft')
ensemble = ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)
print(ensemble_score)
ensemble_probs = ensemble.predict_proba(X_test)
ensemble_probs = ensemble_probs[:, 1]
# print(NB_probs)
ensemble_auc = roc_auc_score(y_test, ensemble_probs)
print(ensemble_auc)
ensemble_auc = round(ensemble_auc, 4)
print(ensemble_auc)

ensemble_fpr, ensemble_tpr, thresholds = roc_curve(y_test, ensemble_probs, pos_label=1)

# ensemble model plot
plt.plot(ensemble_fpr, ensemble_tpr, color='orange', label='ensemble_AUC = {}'.format(ensemble_auc))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()






