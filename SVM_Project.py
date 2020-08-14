# Importing the libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt



dataset = np.loadtxt("/Users/abhishekshah/Desktop/Python_MasterClass/DataMining-Project-master/abalone.csv", delimiter=",")
X = dataset[: , 1:9]
y = dataset[:, 0]


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)

# Training the Kernel SVM model on the Training set

classifier = SVC( C= 20, kernel ='rbf', random_state = 0, probability= True)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)


# ROC Curve plot
prob = classifier.predict_proba(X_test)
prob = prob[:, 1]
auc = metrics.roc_auc_score(y_test, prob)
print('AUC: {}\n'.format(auc))
fpr, tpr, thresholds = metrics.roc_curve(y_test, prob)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = {}'.format(auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Confusion Matrix
print('SVM Confusion Matrix')
print('-------------------------')
print(confusion_matrix(y_test, y_pred))
