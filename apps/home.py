import streamlit as st

def app():
    st.title('DWDM Mini Project')
    st.markdown(
'''
### Title : Comparative Study of Classification Algorithms
### Team :

Name|Reg No.
---|---
Nishan D'Almeida|180911162
Dipankar Srirag|180911176
Yash Wardhan|180911178

### Introduction:
A comparative Study on three different Classification Algorithms namely - `Naive Bayes`, `SVM` and `Logistic Regression` using three different datasets to analyse the robustness of each algorithm and making an educated decision on the type of algorithm to be used on a general classification task. We also try to build a web-app similar to `Weka` or `Rapid Miner`

### Problem Definition:
In day-to-day life data surrounds us. To derive insights from these data various Data Mining Techniques such as Classification can be used. There is an availability of a large number of Classification Algorithms which leads to questions like which algorithm yields better results for the task in hand.

### Objectives:
- Determining an appropriate Classification Algorithm for different types of dataset based on evidence / results generated from the analysis.
- Analysing data and deriving insights on influence of features on the Classification tasks.

### Technology / Algorithms used:
The Study is implemented in Python programming language, further using Python libraries like `NumPy` for operations on datasets and implementation of Algorithms - `Naive Bayes`, `SVM`, `Linear Regression`, `Seaborn` for Data Visualization and Exploratory Analysis as well as `Pandas` for handling the Datasets. `Streamlit` to build a webapp for deployment.

### Methodology:
The Study starts with the collection of three different datasets. These datasets are further processed and exploratory data analysis is to be done, gathering insights such as feature importance from the data which aids us while implementing the algorithms. The algorithms are implemented using `NumPy`, a Python library for array operations and Statistical tasks. For the analysis of the algorithms and their comparison, the data is split into training and testing sets of `7:3` ratio. After the models are built and trained on the processed data we use metrics such as `Accuracy Score` to compare the performance of each algorithm on the dataset and determine the most suitable algorithm for each type of dataset. With `Streamlit` we build a webapp to deploy the app and also introduce dynamic behaviour to the project where the user can have a range of choices from datasets to model to be used for prediction.

### Expected Result:
An appropriate Classification Algorithm which yields better results on a general Classification task.
'''
)


