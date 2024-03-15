import pandas as pd

df = pd.read_csv('spam.csv')
print(df.head())

# this will give you the amount of data in each columns:
print(df.groupby('Category').describe())

# Now, converting the text into numbers:
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head())

# Traning and testing the data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

# conveting the MESSAGE TEXT into the NUMBERS using the COUNTVECTORIZER:
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
print(X_train_count)

# Naive bayes model:
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

# Training the model:
model.fit(X_train_count, y_train)
# Accuracy of the model:
X_test_count = v.transform(X_test)
print(model.score(X_test_count, y_test))
print(model.predict(X_test_count[0:30]))

# we use Pipeline for the same as we do CountVectorizer in the 18-32 lines
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
# training the model using the clf:
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict(X_test[0:30]))


'''
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()

print(dir(wine))
print(wine.feature_names )
print(wine.data[0:2])

# Conveting the iris datas into the dataset:
df = pd.DataFrame(wine.data, columns=wine.feature_names)
print(df.head())

df['target'] = wine.target
print(df[50:70])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
G_model = GaussianNB()
G_model.fit(X_train, y_train)
print(G_model.score(X_test, y_test))
M_model = MultinomialNB()
M_model.fit(X_train, y_train)
print(M_model.score(X_test, y_test))
'''