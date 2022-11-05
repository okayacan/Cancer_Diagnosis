# Classification of cancer diagnosis
# First we import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# We import the dataset. You can get it from Kaggle website. 
df = pd.read_csv('data_cancer.csv')
df.head()

print("Cancer data set dimensions : {}".format(df.shape))

# We look csv file in detail.
df.describe()

# We check the name of columns.
df.columns

# We check the size of csv file.
df.shape

# We find out that there is a full-NaN column called "Unnamed: 32"
df.isna().sum()

# We drop 'id' and 'Unnamed: 32' columns from dataframe.
df.drop(['id'], axis = 1, inplace=True)
df.drop(['Unnamed: 32'], axis = 1, inplace=True)

# We check the new columns.
df.columns

# We group the dataframe by 'diagnosis'. We find out the amounts of B (benign) and M (malign) data.
df.groupby('diagnosis').size()

# We Visualize the dataframe.
df.groupby('diagnosis').hist(figsize=(14, 14))

# We define X and Y variables.
X = df.drop(columns = 'diagnosis')
Y = df.diagnosis


#Encoding categorical data values

# We call LabelEncoder to label Y variable.
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# We Split the data set into the Training and Test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 101)


# Now we start feature Scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# We define a function named 'train_val' to calculate scores.
# 'model' refers to Logistic Regression that we call later.
def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# We try to Fit the Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 101)
classifier.fit(X_train, y_train)


# Now we predict for the Test data set.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
train_val(classifier, X_train, y_train, X_test, y_test)


# We define a function for calculating the adjusted r2 parameter value.
def adj_r2(y_test, y_pred, df):
    r2 = r2_score(y_test, y_pred)
    n = df.shape[0]   # number of observations
    p = df.shape[1]-1 # number of independent variables 
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2


# We calculate Y value for the test data set.
y_pred = classifier.predict(X_test)

# We calculate the adjusted r2 score.
adj_r2(y_test, y_pred, df)


## Cross Validation

# We call the relevant libraries for cross validation process.
# Secondly, we define our model (Logistic regression). We set CV as 5.
# Finally we calculate the scores.
from sklearn.model_selection import cross_validate, cross_val_score
model = classifier 
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv = 5)

# We list the scores as dataframe.
pd.DataFrame(scores)

# Now we calculate the mean values of scores over cv value.
pd.DataFrame(scores).iloc[:, 2:].mean()


# We prepare a classification report.
from yellowbrick.classifier import ClassificationReport

visualizer = ClassificationReport(model) # Our model is Logistic Regression as defined above.

visualizer.fit(X_train, y_train)  # We fit the training data to the visualizer
visualizer.score(X_test, y_test)  # We evaluate the model on the test data
visualizer.show();


# We find the best alpha value based on Ridge model using cross validation.
from sklearn.linear_model import RidgeCV
from yellowbrick.regressor import AlphaSelection

alphas = np.logspace(-10, 1, 200)
visualizer = AlphaSelection(RidgeCV(alphas=alphas))
visualizer.fit(X, Y)
visualizer.show()


# We calculate confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_pred, model.predict(X_test))


# We first get dummies for diagnosis feature (i.e. our dependent variable).
df = pd.get_dummies(df, drop_first =True)


# We see the new data frame.
# We see that diagnosis changed to diagnosis_M.
df.head()


# We calculate correlation parameters for dependent variable diagnosis_M.
corr_by_diag = df.corr()["diagnosis_M"].sort_values()[:-1]
corr_by_diag

# We plot correlate/uncorrelated variables for diagnosis_M.
plt.figure(figsize = (15,8))
sns.barplot(x = corr_by_diag.index, y = corr_by_diag)
plt.xticks(rotation=90)
plt.tight_layout()

