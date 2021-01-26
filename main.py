import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pydotplus


def main():
    # Loads the data into panda data frame
    penguinsData = pd.read_csv("penguins.csv")

    # drop null or not number rows, convert categorical features to numerical
    penguinsData = penguinsData.dropna()
    penguinsData.sex = pd.get_dummies(penguinsData.sex)
    penguinsData.island = pd.Categorical(penguinsData.island,
                                              categories=penguinsData.island.unique()).codes
    penguinsData.species = pd.Categorical(penguinsData.species,
                                               categories=penguinsData.species.unique()).codes
    # print(penguinsData)
    # Task 1.1
    # "1. Select the 2 features which allow for the most accurate 2-feature GNB classifier. Explain your selection."
    penguins = penguinsData
    penguins = sns.load_dataset('penguins')
    sns.pairplot(penguins, hue='species', height=1.5)
    plt.show()
    # We can see that we need the features bill_depth_mm and bill_length_mm there is the more separate between species.

    # Task 1.2
    # 2. Train your model using 80% of the data set as your training set.
    penguins = penguinsData
    X_penguins = penguins.drop(['species'], axis=1)
    y_penguins = penguins['species']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_penguins, y_penguins, test_size=0.2, random_state=1)
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    ypred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([Xtest.reset_index(), ytest.reset_index(), ypred], axis=1)
    print(metrics.accuracy_score(ytest, y_model))
    print(predicted)

    # Task 1.3
    # 3. Use a filled contour plot to show the decision distribution of your model (limit your plot axes to\n",
    #     "the actual data boundaries +-1).


    # print(penguinsData)

if __name__ == '__main__':
    main()
