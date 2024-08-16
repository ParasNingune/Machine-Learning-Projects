from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree

def main():
    print("-----------------IRIS FLOWER CLASSIFICATION-----------------")

    # Load the iris dataset
    iris = load_iris()

    #print(iris)
    # Extract the features (X) and target variable (y) from the dataset
    X = iris.data
    y = iris.target

    #print(X)# Split the data into training and testing sets (75% for training and 25% for testing)
    data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.75)

    # Create a Decision Tree Classifier object
    obj = tree.DecisionTreeClassifier()
    # Train the model using the training data
    obj.fit(data_train, target_train)

    # Use the trained model to make predictions on the testing data
    output_obj = obj.predict(data_test)
    print(output_obj)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(target_test, output_obj)

    print("Accuracy: ", accuracy*100, "")
    #tree.plot_tree(obj)
    

if __name__ == "__main__":
    main()