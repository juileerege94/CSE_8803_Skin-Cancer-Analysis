# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:59:21 2017

@author: Juilee Rege
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pickle
import cv2

with open('./preprocessed_data/features_v2', 'rb') as handle:
    u = pickle._Unpickler(handle)
    u.encoding = 'latin1'
    features = u.load()
with open('./preprocessed_data/names_v2', 'rb') as handle:
    u = pickle._Unpickler(handle)
    u.encoding = 'latin1'
    names = u.load()

count1_rf = 0
count2_rf = 0
count3_rf = 0
count1_svc = 0
count2_svc = 0
count3_svc = 0

X_train, X_test, l_train, l_test = train_test_split(features, names, test_size=0.1, random_state=42)
y_train = [int(x[0]) for x in l_train]
y_test =  [int(x[0]) for x in l_test]

###############################  Random Forests on Preprocessed Data  ##########################################

M1 = RandomForestClassifier(n_estimators = 10, criterion='entropy')
M1.fit(X_train,y_train)
M2 = RandomForestClassifier(n_estimators = 50, criterion='entropy')
M2.fit(X_train,y_train)
M3 = RandomForestClassifier(n_estimators = 200, criterion='entropy')
M3.fit(X_train,y_train)
print("Finish training!")

predicted1_rf = M1.predict_proba(X_test)
y_pred1_rf = M1.predict(X_test)
y_pred_score1_rf = predicted1_rf[:,1].tolist()
fpr1_rf, tpr1_rf, thresholds1_rf = roc_curve(y_test, y_pred_score1_rf)
roc_auc1_rf = auc(fpr1_rf,tpr1_rf)

predicted2_rf = M2.predict_proba(X_test)
y_pred2_rf = M2.predict(X_test)
y_pred_score2_rf = predicted2_rf[:,1].tolist()
fpr2_rf, tpr2_rf, thresholds2_rf = roc_curve(y_test, y_pred_score2_rf)
roc_auc2_rf = auc(fpr2_rf,tpr2_rf)

predicted3_rf = M3.predict_proba(X_test)
y_pred3_rf = M3.predict(X_test)
y_pred_score3_rf = predicted3_rf[:,1].tolist()
fpr3_rf, tpr3_rf, thresholds3_rf = roc_curve(y_test, y_pred_score3_rf)
roc_auc3_rf = auc(fpr3_rf,tpr3_rf)
cf = confusion_matrix(y_test, y_pred3_rf)

for i, j in zip(y_pred1_rf, y_test):
    if(i!=j):
        count1_rf+=1
for i, j in zip(y_pred2_rf, y_test):
    if(i!=j):
        count2_rf+=1
for i, j in zip(y_pred3_rf, y_test):
    if(i!=j):
        count3_rf+=1

for i in range(len(y_pred3_rf)):
    if(y_pred3_rf[i]!=y_test[i]):
        count3_rf+=1
        img = cv2.imread('./preprocessed_data/Data_V2/'+l_test[i]+".jpg")
        cv2.imwrite('./preprocessed_data/misclassified/'+ l_test[i]+".jpg", img)
    else:
        img = cv2.imread('./preprocessed_data/Data_V2/'+l_test[i]+".jpg")
        cv2.imwrite('./preprocessed_data/correctly_classified/'+ l_test[i]+".jpg", img)
        
print ("RF n = 10")
print("misclassified", count1_rf)
print("total validation", len(y_test))
print ("")
print ("RF n = 50")
print("misclassified", count2_rf)
print("total validation", len(y_test))
print ("")
print ("RF n = 200")
print("misclassified", count3_rf)
print("total validation", len(y_test))
print ("")

plt.figure()
plt.plot(fpr1_rf, tpr1_rf, '-', lw=2)
plt.plot(fpr2_rf, tpr2_rf, '-', lw=2)
plt.plot(fpr3_rf, tpr3_rf, '-', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for pre-processed Random Forests')
plt.legend(['n = 10 (area = %0.3f)' % roc_auc1_rf, 'n = 50 (area = %0.3f)' % roc_auc2_rf, 'n = 200 (area = %0.3f)' % roc_auc3_rf], loc='best')
plt.show()

labels = ['benign', 'malignant']
plt.figure(figsize=(5, 5))
plt.imshow(cf, cmap = 'PuBu')
for y in range(2):
    for x in range(2):
        plt.text(x , y, cf[y][x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.xticks(range(len(cf)), labels, size='small')
plt.yticks(range(len(cf)), labels, size='small')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title('Confusion Matrix')
plt.show()
print ("Accuracy of Classification")
print (accuracy_score(y_test,y_pred3_rf))

TN = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TP = cf[1][1]

sensitivity = float(TP / (TP + FN))
specificity = float(TN / (TN + FP))
PPV = float(TP / (TP + FP))
NPV = float(TN / (TN + FN))

print ("Sensitivity: ",sensitivity)
print ("Specificity: ",specificity)
print ("PPV: ",PPV)
print ("NPV:" , NPV)

###############################  Linear SVC on Preprocessed Data  ##########################################

clf1 = LinearSVC(C=0.7, loss='squared_hinge', penalty='l2')
clf1.fit(X_train, y_train)
clf2 = LinearSVC(C=1, loss='squared_hinge', penalty='l2')
clf2.fit(X_train, y_train)
clf3 = LinearSVC(C=2, loss='squared_hinge', penalty='l2')
clf3.fit(X_train, y_train)
print ("Finished Training")

y_pred1_svc = clf1.predict(X_test)
y_pred_score1_svc = clf1.decision_function(X_test)
cf = confusion_matrix(y_test, y_pred1_svc)

y_pred2_svc = clf2.predict(X_test)
y_pred_score2_svc = clf2.decision_function(X_test)
y_pred3_svc = clf3.predict(X_test)
y_pred_score3_svc = clf3.decision_function(X_test)

fpr1_svc, tpr1_svc, thresholds1_svc = metrics.roc_curve(y_test, y_pred_score1_svc)
roc_auc1_svc = auc(fpr1_svc,tpr1_svc)
fpr2_svc, tpr2_svc, thresholds2_svc = metrics.roc_curve(y_test, y_pred_score2_svc)
roc_auc2_svc = auc(fpr2_svc,tpr2_svc)
fpr3_svc, tpr3_svc, thresholds3_svc = metrics.roc_curve(y_test, y_pred_score3_svc)
roc_auc3_svc = auc(fpr3_svc,tpr3_svc)

for i, j in zip(y_pred1_svc, y_test):
    if(i!=j):
        count1_svc+=1
for i, j in zip(y_pred2_svc, y_test):
    if(i!=j):
        count2_svc+=1
for i, j in zip(y_pred3_svc, y_test):
    if(i!=j):
        count3_svc+=1
        
print ("SVC c = 0.7")
print("misclassified", count1_svc)
print("total validation", len(y_test))
print ("")
print ("SVC c = 1")
print("misclassified", count2_svc)
print("total validation", len(y_test))
print ("")
print ("SVC c = 2")
print("misclassified", count3_svc)
print("total validation", len(y_test))
print ("")

plt.figure()
plt.plot(fpr1_svc, tpr1_svc, '-', lw=2)
plt.plot(fpr2_svc, tpr2_svc, '-', lw=2)
plt.plot(fpr3_svc, tpr3_svc, '-', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for pre-processed Linear SVC')
plt.legend(['c = 0.7 (area = %0.3f)' % roc_auc1_svc, 'c = 1 (area = %0.3f)' % roc_auc2_svc, 'c = 2 (area = %0.3f)' % roc_auc3_svc], loc='best')
plt.show()

labels = ['benign', 'malignant']
plt.figure(figsize=(5, 5))
plt.imshow(cf, cmap = 'PuBu')
for y in range(2):
    for x in range(2):
        plt.text(x , y, cf[y][x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
plt.xticks(range(len(cf)), labels, size='small')
plt.yticks(range(len(cf)), labels, size='small')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title('Confusion Matrix')
plt.show()
print ("Accuracy of Classification")
print (accuracy_score(y_test,y_pred3_rf))

TN = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TP = cf[1][1]

sensitivity = float(TP / (TP + FN))
specificity = float(TN / (TN + FP))
PPV = float(TP / (TP + FP))
NPV = float(TN / (TN + FN))

print ("Sensitivity: ",sensitivity)
print ("Specificity: ",specificity)
print ("PPV: ",PPV)
print ("NPV:" , NPV)


#########################  Best Linear SVC & Best RF on Preprocessed Data  #####################################

M1 = RandomForestClassifier(n_estimators = 200, criterion='entropy')
M1.fit(X_train,y_train)
predicted1_rf = M1.predict_proba(X_test)
y_pred1_rf = M1.predict(X_test)
y_pred_score1_rf = predicted1_rf[:,1].tolist()
fpr1_rf, tpr1_rf, thresholds1_rf = roc_curve(y_test, y_pred_score1_rf)
roc_auc1_rf = auc(fpr1_rf,tpr1_rf)
for i, j in zip(y_pred1_rf, y_test):
    if(i!=j):
        count1_rf+=1
print ("RF n = 200")
print("misclassified", count1_rf)
print("total validation", len(y_test))
print ("")


clf1 = LinearSVC(C=0.7, loss='squared_hinge', penalty='l2')
clf1.fit(X_train, y_train)
y_pred1_svc = clf1.predict(X_test)
y_pred_score1_svc = clf1.decision_function(X_test)
fpr1_svc, tpr1_svc, thresholds1_svc = metrics.roc_curve(y_test, y_pred_score1_svc)
roc_auc1_svc = auc(fpr1_svc,tpr1_svc)
for i, j in zip(y_pred1_svc, y_test):
    if(i!=j):
        count1_svc+=1
print ("SVC c = 0.7")
print("misclassified", count1_svc)
print("total validation", len(y_test))
print ("")

plt.figure()
plt.plot(fpr1_rf, tpr1_rf, '-', lw=2)
plt.plot(fpr1_svc, tpr1_svc, '-', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for best SVC and best RF')
plt.legend(['n = 200 (area = %0.3f)' % roc_auc1_rf, 'c = 0.7 (area = %0.3f)' % roc_auc1_svc], loc='best')
plt.show()