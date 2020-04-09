# Predicting Survivors for Titanic
import pandas as pd
import utils
from sklearn import tree

# Preparing dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
utils.clean_data(train)
utils.clean_data(test)

# Training model
target = train["Survived"].values
features_names = ["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]
features = train[features_names].values

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree_ = decision_tree.fit(features, target)

print (decision_tree_.score(features, target))

features_test = test[features_names].values
survived = decision_tree_.predict(features_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": survived
        })

submission.to_csv('submission.csv', index = False)

