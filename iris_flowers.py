#load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
print("\n\n")

# head
print(dataset.head(20))
print("\n\n")

# descriptions
print(dataset.describe())
print("\n\n")

# class distribution
print(dataset.groupby('class').size())
print("\n\n")

# histograms
dataset.hist()
plt.show()

#seperate the validation set
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
# using seed so that random splitting of the dataset is same for all the algorithms
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# test options and evaluation metric
seed = 7
scoring = 'accuracy'

# using 6 different types of algorithms to see which one has the highest accuracy, mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms
#spot chaeck algorithms
models  = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate each model in turn
results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)
    print("\n\n")


# the knn algorithm was an appropriate model based on our tests
# make predictions for the knn algorithm on the validation set
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print("\n\n")
# confusion matrix is the table of true positives,true negatives,false positives and false negatives
print(confusion_matrix(Y_validation,predictions))
print("\n\n")
print(classification_report(Y_validation,predictions))
print("\n\n")
