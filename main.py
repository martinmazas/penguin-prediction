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
# pd.set_option("display.max_rows", None, "display.max_columns", None)


def get_wrong_predictions(feature1, feature2, data, spread=1):
    x = pd.concat([data[feature1], data[feature2]], axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

    clf = GaussianNB()
    # if (model != "gnb"):
    #     clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)

    # Train Classifer

    prob = len(clf.classes_) == 2

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    x_min, x_max = x.loc[:, feature1].min() - 1, x.loc[:, feature1].max() + 1
    y_min, y_max = x.loc[:, feature2].min() - 1, x.loc[:, feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    if prob:
        Z = Z[:, 1] - Z[:, 0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_) + 3)

    prediction = clf.predict(x)
    predicted = pd.DataFrame(prediction, columns=['predicted'])
    data_with_prediction = pd.concat([x, y, predicted], axis=1)
    data_with_wrong_predictions = data_with_prediction.where(data_with_prediction['species'] != data_with_prediction['predicted'])
    data_with_wrong_predictions = data_with_wrong_predictions.dropna()

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)

    if not prob:
        plt.clim(0, len(clf.classes_) + 3)

    hueorder = clf.classes_
    sns.scatterplot(data=data_with_wrong_predictions[::spread], x=feature1, y=feature2, hue=y, hue_order=hueorder, palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


def two_features_plot(feature1, feature2, data, species):
    penguins = data
    concat_features = pd.concat([feature1, feature2], axis=1)
    X_penguins = concat_features
    y_penguins = penguins['species']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_penguins, y_penguins, random_state=1)
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    ypred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([Xtest.reset_index(), ytest.reset_index(), ypred], axis=1)
    plt.scatter(feature1, feature2, alpha=0.6, c=y_penguins, cmap='jet')
    sns.scatterplot(data=data, x=feature1, y=feature2, hue=species)
    plt.xlabel(feature1.name)
    plt.ylabel(feature2.name)
    plt.show()
    print(metrics.accuracy_score(ytest, y_model))


def main():
    # Loads the data into panda data frame
    penguinsData = pd.read_csv("penguins.csv")

    species = penguinsData['species']

    # drop null or not number rows, convert categorical features to numerical
    # penguinsData = penguinsData.dropna()
    penguinsData.fillna(penguinsData.mean(), inplace=True)
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
    # two_features_plot(penguinsData['bill_length_mm'], penguinsData['body_mass_g'], penguinsData, species)
    # two_features_plot(penguinsData['bill_length_mm'], penguinsData['flipper_length_mm'], penguinsData, species)
    # two_features_plot(penguinsData['bill_length_mm'], penguinsData['bill_depth_mm'], penguinsData, species)



    # We can see that we need the features bill_depth_mm and bill_length_mm there is the more separate between species.

    # Task 1.2
    #2. Train your model using 80% of the data set as your training set.
    X_penguins = penguinsData.drop(['species'], axis=1)
    y_penguins = penguinsData['species']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_penguins, y_penguins, test_size=0.2, random_state=1)
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    ypred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([Xtest.reset_index(), ytest.reset_index(), ypred], axis=1)
    # print(metrics.accuracy_score(ytest, y_model))
    # print(predicted)
    #
    # # Task 1.3
    # # 3. Use a filled contour plot to show the decision distribution of your model (limit your plot axes to\n",
    # #     "the actual data boundaries +-1).

    def bayes_plot(df, model="gnb", spread=30):
        df.dropna()
        colors = 'seismic'
        col1 = df.columns[0]
        col2 = df.columns[1]
        target = df.columns[2]
        sns.scatterplot(data=df, x=col1, y=col2, hue=target)
        plt.show()
        y = df[target]  # Target variable
        X = df.drop(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=1)  # 80% training and 20% test

        clf = GaussianNB()
        if (model != "gnb"):
            clf = DecisionTreeClassifier(max_depth=model)
        clf = clf.fit(X_train, y_train)

        # Train Classifer

        prob = len(clf.classes_) == 2

        # Predict the response for test dataset

        y_pred = clf.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))

        hueorder = clf.classes_

        def numify(val):
            return np.where(clf.classes_ == val)[0]

        Y = y.apply(numify)

        x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
        y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))

        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if prob:
            Z = Z[:, 1] - Z[:, 0]
        else:
            colors = "Set1"
            Z = np.argmax(Z, axis=1)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
        plt.colorbar()
        if not prob:
            plt.clim(0, len(clf.classes_) + 3)

        sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder, palette=colors)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.show()


    # input_values = pd.concat([penguinsData.bill_depth_mm, penguinsData.bill_length_mm], axis=1)
    # target_values = penguins.species
    # bayes_plot(pd.concat([input_values, target_values], axis=1), spread=1)

    input_values = pd.concat([penguinsData.bill_depth_mm, penguinsData.bill_length_mm], axis=1)
    target_values = penguins.species
    # bayes_plot(pd.concat([input_values, target_values], axis=1), spread=1)

    # print(penguinsData)

    get_wrong_predictions('bill_depth_mm', 'bill_length_mm', penguinsData)

    # penguins.fillna(penguinsData.mean(), inplace=True)
    # print(penguins)
    conditions = [0, 1, 2]
    choices = ['Adelie', 'Chinstrap', 'Gentoo']
    spec = pd.DataFrame(np.where(penguinsData['species'] == 0, 'Adelie',
                                     (np.where(penguinsData['species'] == 1, 'Chinstrap', 'Gentoo'))), columns=['spec'])
    sex = pd.DataFrame(np.where(penguinsData['sex'] == 0, 'Male', "Female"), columns=['sex'])

    new_col = pd.concat([spec, sex], axis=1)
    new_col['class'] = new_col['sex'] + " " + new_col['spec']
    penguinsData = pd.concat([penguinsData, new_col['class']], axis=1)
    # print(penguinsData)

    penguinsData['class'] = pd.Categorical(penguinsData['class'],
                                          categories=penguinsData['class'].unique()).codes
    # penguinsData.to_csv('newPenguins.csv',index=None)
    penguins['class'] = penguins['sex'] + " " + penguins['species']


if __name__ == '__main__':
    main()
