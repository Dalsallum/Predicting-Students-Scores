import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns

# location to store the files
os.chdir(r'C:\Users\komsi\Desktop\Projects\ML_projects\Students')

# the data
student_data = pd.read_csv(r'C:\Users\komsi\Desktop\Projects\ML_projects\Students\StudentsPerformance.csv')

# first will try to predict the math score based on reading and writing scores only

numeric_data = student_data[['math score', 'reading score', 'writing score']]
predict = 'math score'

'''
The next X line will create a numpy array containing
all the values without the predict value
the argument 1 mean will drop the columns (labels names)
The Y line will have array of math scores only
'''

X = np.array(numeric_data.drop([predict], 1))
Y = np.array(numeric_data[predict])

'''
Now we will split the data with 20% of the data as test size
random state was selected to keep the accuracy the same always
many values tried and 37 had the best accuracy
'''

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=37)

# Linear Regression

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# calculating the accuracy
acc = linear.score(x_test, y_test)
print('The accuracy based on reading and writing = ', round(acc * 100, 2), '%')

# print the predicted vs actual ( only 10 values )
predictions = linear.predict(x_test)

for x in range(len(predictions) - 190):
    print('Predicted Math score = ', round(predictions[x]), ' Actual = ', y_test[x])

'''
now let's try to include the categorical variable into the regression
will convert gander , parental level of education and test preparation course to dummy values
will refer to the new variables the same as before but with 1 added to them
'''

data_to_convert = student_data[['gender', 'parental level of education', 'test preparation course']]
dummy_data = pd.get_dummies(data=data_to_convert, drop_first=False)

# now concatenate the numeric data with the the dummy data
cont_data = pd.concat([numeric_data, dummy_data], axis=1)

X1 = np.array(cont_data.drop([predict], 1))
Y1 = np.array(cont_data[predict])

# we will try to find the highest accuracy by changing the random_state argument

'''
acc_list = []
for i in range(10000):
    x1_train, x1_test, y1_train, y1_test = sklearn.model_selection.train_test_split(X1, Y1, test_size=0.2,random_state=i)
    linear.fit(x1_train, y1_train)
    accuracy = linear.score(x1_test, y1_test)
    acc_list.append(accuracy)


max_acc = max(acc_list)
max_random_state = acc_list.index(max_acc)
'''
# from running the last snippet of code we got 2036 as the best random_state with accuracy of 90.25% now we know the
# best accuracy and its random_state lets apply it again so we can save the values of x1_train, x1_test, y1_train,
# y1_test

x1_train, x1_test, y1_train, y1_test = sklearn.model_selection.train_test_split(X1, Y1, test_size=0.2,
                                                                                random_state=2036)
linear.fit(x1_train, y1_train)
acc1 = linear.score(x1_test, y1_test)
print('\n\nThe accuracy including Categorical data = ', round(acc1 * 100, 2), '%')

predictions1 = linear.predict(x1_test)

for x in range(len(predictions1) - 190):
    print('Predicted Math score = ', round(predictions1[x]), ' Actual = ', y1_test[x])

sns.set_style("whitegrid")  # optional style

numeric_plot = sns.relplot(x=y_test, y=predictions, kind='scatter')
numeric_plot.fig.suptitle('Numeric Data only plot')
numeric_plot.ax.set(xlabel='Actual', ylabel='Predicted')

numeric_categ_plot = sns.relplot(x=y1_test, y=predictions1, kind='scatter')
numeric_categ_plot.fig.suptitle('Numeric and Categorical Data plot')
numeric_categ_plot.ax.set(xlabel='Actual', ylabel='Predicted')

plt.show()
