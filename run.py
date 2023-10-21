import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report, accuracy_score ,confusion_matrix
#create dataframe from csv
df = pd.read_csv('spam_ham_dataset.csv')
df.head()
print("spam count: " +str(len(df.loc[df.label_num==1])))
print("not spam count: " +str(len(df.loc[df.label_num==0])))
print(df.shape)
df['label_num'] = df['label_num'].astype(int)
df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['text','label_num']]
print(df.shape)


#transform the text (feauter extraction)
text_vec = CountVectorizer().fit_transform(df['text'])
# split data set to Training and Testing data-set  by (80%,20%)
X_train, X_test, y_train, y_test = train_test_split(text_vec, df['label_num'], test_size = 0.20, shuffle = True)
# classifier choice is  Gradient Boosting
classifier = ensemble.GradientBoostingClassifier(
    n_estimators = 100, #how many decision trees to build
    learning_rate = 0.5, #learning rate
    max_depth = 6
)
# Training by Gradient Boosting classifier
classifier.fit(X_train, y_train)
# prediction by Gradient Boosting classifier used Testing data-set
predictions = classifier.predict(X_test)
# print the confusion matrix
print(confusion_matrix(y_test, predictions))
# print the report of the prediction
print(classification_report(y_test, predictions))
# classifier choice is  Random Forest
classifier = ensemble.RandomForestClassifier(
    n_estimators = 100, #how many decision trees to build
    #learning_rate = 0.5, #learning rate
    max_depth = 6
)
# Training by  Random Forest
classifier.fit(X_train, y_train)
# prediction by Gradient Boosting classifier used Testing data-set
predictions = classifier.predict(X_test)
# print the confusion matrix
print(confusion_matrix(y_test, predictions))
# print the report of the prediction
print(classification_report(y_test, predictions))