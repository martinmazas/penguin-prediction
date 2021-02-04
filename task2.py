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
    y = data['class']
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
    data_with_wrong_predictions = data_with_prediction.where(data_with_prediction['class'] != data_with_prediction['predicted'])
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


def two_features_plot(feature1, feature2, data, classes):
    penguins = data
    concat_features = pd.concat([feature1, feature2], axis=1)
    X_penguins = concat_features
    y_penguins = penguins['class']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_penguins, y_penguins, random_state=1)
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    ypred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([Xtest.reset_index(), ytest.reset_index(), ypred], axis=1)
    plt.scatter(feature1, feature2, alpha=0.6, c=y_penguins, cmap='jet')
    sns.scatterplot(data=data, x=feature1, y=feature2, hue=classes)
    plt.xlabel(feature1.name)
    plt.ylabel(feature2.name)
    plt.show()
    print(metrics.accuracy_score(ytest, y_model))


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


def main():
    penguins_data = pd.read_csv("penguins.csv")

    penguins_data.sex = pd.get_dummies(penguins_data.sex)
    penguins_data.island = pd.Categorical(penguins_data.island,
                                          categories=penguins_data.island.unique()).codes
    penguins_data.species = pd.Categorical(penguins_data.species,
                                           categories=penguins_data.species.unique()).codes

    penguins_data.fillna(penguins_data.mean(), inplace=True)

    spec = pd.DataFrame(np.where(penguins_data['species'] == 0, 'Adelie',
                                 (np.where(penguins_data['species'] == 1, 'Chinstrap', 'Gentoo'))), columns=['spec'])
    sex = pd.DataFrame(np.where(penguins_data['sex'] == 0, 'Male', "Female"), columns=['sex'])

    new_col = pd.concat([spec, sex], axis=1)
    new_col['class'] = new_col['sex'] + " " + new_col['spec']
    penguins_data = pd.concat([penguins_data, new_col['class']], axis=1)

    sns.pairplot(penguins_data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'class']],
                 hue='class', height=1.5)
    plt.show()

    classes = penguins_data['class']
    penguins_data['class'] = pd.Categorical(penguins_data['class'],
                                            categories=penguins_data['class'].unique()).codes
    # two_features_plot(penguins_data['bill_length_mm'], penguins_data['body_mass_g'], penguins_data, classes)
    # two_features_plot(penguins_data['bill_length_mm'], penguins_data['flipper_length_mm'], penguins_data, classes)
    # two_features_plot(penguins_data['bill_length_mm'], penguins_data['bill_depth_mm'], penguins_data, classes)

    X_penguins = penguins_data.drop(['class', 'sex', 'species'], axis=1)
    y_penguins = penguins_data['class']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_penguins, y_penguins, test_size=0.2, random_state=1)
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    ypred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([Xtest.reset_index(), ytest.reset_index(), ypred], axis=1)
    # print(metrics.accuracy_score(ytest, y_model))
    # print(predicted)

    penguins_data['class'] = pd.DataFrame(np.where(penguins_data['class'] == 0, 'Male Adelie',
                                 (np.where(penguins_data['class'] == 1, 'Female Adelie',
                                (np.where(penguins_data['class'] == 2,'Female Chinstrap',
                                (np.where(penguins_data['class'] == 3, 'Male Chinstrap',
                                (np.where(penguins_data['class'] == 4, 'Female Gentoo', 'Male Gentoo'))))))))), columns=['spec'])
    input_values = pd.concat([penguins_data.bill_depth_mm, penguins_data.bill_length_mm], axis=1)
    target_values = penguins_data['class']
    bayes_plot(pd.concat([input_values, target_values], axis=1), spread=1)

    get_wrong_predictions('bill_depth_mm', 'bill_length_mm', penguins_data)


if __name__ == "__main__":
    main()
