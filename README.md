
# Police Challan

The Dataset comprises of the vehicles that 
were stopped at a Police Checkpost, which 
includes all the basic data that a basic checkpost
assesment goes through.

## About the Dataset

The diferent Catagories of the dataset are
comprised of the Gender, Date, Age, Violation,
stop duration, drug related stop

Ultimately the dataset gives us the information
according to which we will finally predict, 
whether the person is arrested or not. 
So in order
to acknowledge the final prediction we require all the 
possible values for which the person could be questioned, 
therefore we will be toggling with the dataset to get
the best possible dependent variables, so that
we get the highest accuracy of prediction.
## Data Analysis Tools

### 1. Simple Imputer:

The Dataset contained ertain Missing Values in it, for which I used the most common Data Preprocessing tool of Machine Learning library that is Simple Imputer, and replaced the missing values with the 'Most Frequent Values'.
https://user-images.githubusercontent.com/109500969/179671707-703f2183-92a1-4103-b0cf-3927173d146d.png



### 2. Label Encoder

In order to run Machine Learning algorithms,
we need to encode all the catagorical data, so
that the model can analyse it better, and all the 
values are stored in the form of matrix of features and makes it
a way more easier to apply machine learning algorithms.
## Deployment

To deploy this project run

```bash
# Police Dataset Toggling

## Importing Libraried and Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv('/content/3. Police Data.csv')

df.drop(columns=['violation_raw', 'driver_age_raw', 'search_type', 'driver_race', 'stop_date', 'stop_time','country_name'], inplace=True)
df.head(2)

##Removing the NULL values from the Dataset

df.isnull().sum()

cols = ['driver_gender',
       'driver_age', 'violation', 'stop_outcome', 'stop_duration', 'is_arrested']
si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for columns in cols:
  df[columns] = si.fit_transform(df[columns].values.reshape(-1,1))

df.isnull().sum()

##In speeding were men or Women called more often

result = df[df.violation == 'Speeding'].driver_gender.value_counts()
result

sns.countplot(df['driver_gender'])

##Grouping a specific columns, stoping of males were more or Females

 male_vs_female = df.groupby('search_conducted').driver_gender.value_counts()
 male_vs_female

females_stoped = df[df.driver_gender == 'F'].search_conducted.value_counts()
males_stoped = df[df.driver_gender == 'M'].search_conducted.value_counts()
print('Number of females stoped {}'.format(females_stoped))

##What is the mean of stopped Duration by MAPPING Function

df.stop_duration.value_counts()

### Mapping the Values before Mean

df['stop_duration'] = df['stop_duration'].map( {'0-15 Min' : 7.5 , '16-30 Min' : 24 , '30+ Min': 45})
df['stop_duration'].mean()

##Age Distribution for Violation

df.head(2)

df.groupby('driver_age').violation.describe()

# Machine Learning Algorithms

## Encoding the Catagorical Data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols = ['driver_gender', 'violation', 'stop_duration',
       'search_conducted', 'stop_outcome', 'is_arrested',
       'drugs_related_stop']

for columns in cols:
  df[columns] = le.fit_transform(df[columns])

df.head(2)

### Dependent and Independent Variables

x = df.drop(columns='is_arrested').values
y = df['is_arrested'].values

### Test Train Split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

## Defining a common function for algorithm testing:

def classify(x, y):
  classifier.fit(x, y)
  score = accuracy_score(classifier.predict(x_test), y_test)
  print('Accuracy Score for this model is: ', score)

### Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classify(x_train, y_train)

### K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classify(x_train, y_train)

### Decision Tree Classfication

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classify(x_train, y_train)

### Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classify(x_train, y_train)

## Testing our model with input values, which gives whether the person is arrested or not

df.head(2)

input_raw = np.asarray((1, 19, 2, 0, 0, 0, 0))
input = input_raw.reshape(1, -1)
predicted = classifier.predict(input)
print(predicted)

if predicted[0] == 0:
  print('You would not have been rejected')

else:
  print('Your were lucky that day, for not being present')

input_raw = np.asarray((1, 19, 6, 0, 2, 3, 0))
input = input_raw.reshape(1, -1)
predicted = classifier.predict(input)
print(predicted)

if predicted[0] == 0:
  print('You were safe, if u would have been there')

else:
  print('Your were lucky that day, for not being present')

##Conclusion:
###The above two input parameters gives us the detailed decription,  that whether a person would be caught or not, based on the true value of questions provided.
```


## Conclusion

The prediction finally gives us a final idea,
if the particular case senario occurs in front of 
anyone, so would they be arrested or not.

# Hi, I'm Avichal Srivastava! 👋

You can reach out to me at: srivastavaavichal007@gmail.com LinkedIn: www.linkedin.com/in/avichal-srivastava-186865187


## 🚀 About Me
I'm a Mechanical Engineer by education, and love to work with data, I eventually started my coding journey for one of my Drone project, wherein I realized that it is something which makes me feel happy in doing, then
I planed ahead to move in the Buiness Analyst or Data Analyst domain.
The reason for choosing these domains is because I love maths a lot and all the Machine Learning algorithms are completely
based on mathematical intution, So this was about me
Hope! You liked it, and it is just a beginning, many more to come, till then Happy Analysing!!
