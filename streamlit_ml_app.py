import numpy as np
import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA               # principal componant analysis algo
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.title("Streamlit learning")
st.write(""" ## Explore diffirent classifier """)

# selectbox
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):

    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y

def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)

        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params

def  get_classifier(clf_name, params):

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif clf_name == "SVM":
        clf = SVC(C = params['C'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],
                                    max_depth = params['max_depth'],
                                    random_state = 1234
        )

    return clf

X, y = get_dataset(dataset_name)

params = add_parameter_ui(classifier_name)
clf = get_classifier(classifier_name, params)

st.write("Shape of the dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

show_dataset = st.checkbox("Show dataset (Type: {})".format(type(X)))
if show_dataset:
    st.write("{} dataset: ".format(dataset_name), X)
    st.write("Labels: ", y)

# Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# PLOT datasets

pca = PCA(2) # 2 = no. of dimantion (2D), PCA converts features to lowar dimantional space
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principal componet 1")
plt.ylabel("Principal componet 2")
plt.colorbar()

st.pyplot(fig)



# TODO:
## add more parameters (sklearn)
## add other classifiers
## add feature scaling
