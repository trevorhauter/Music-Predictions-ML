import pandas as pd
import sklearn, os
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

#gets the path of the folder the py file is in
path = os.path.dirname(os.path.realpath(__file__)) + "\\"

def getModel(x, y):
    # splits the data sets up in 80/20 train/test sets so we can see how accurate our machine is
    # randomly splits the data so we're going to get different data sets each time
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    modelFile = Path(path + "music-predictor.joblib")

    #checks to see if the model already exists, if not it trains a new one
    if (modelFile).is_file():
        print('Existing model found, loading...')
        #loads the existing model
        model = joblib.load('music-predictor.joblib')
    else:
        print('No model found, creating new model...')
        # creates a new model that we can train with out data sets
        model = DecisionTreeClassifier()
        # trains it on the data we split up for training
        model.fit(xTrain, yTrain)
        # saves the model after training it
        joblib.dump(model, 'music-predictor.joblib')

    return model, xTest, yTest

def start():
    #open the data set as a dataframe
    #there are 3 cols age, gender, genre. Gender is represented by binary numbers, (0 = woman, 1 = man)
    musicData = pd.read_csv('music.csv')

    #creates an input set with the age and gender - because thats whats used to predict the output
    x = musicData.drop(columns=['genre'])
    #creates an output set with only the genre - i.e. the predictions from the input set
    y = musicData['genre']

    #calls a function to either retrieve or create a model - returns our model and test datasets
    model, xTest, yTest = getModel(x, y)

    #uses the data we have for testing and allows the model to make predictions with the remaining data
    predictions = model.predict(xTest)

    #visualizes and exports the decision tree - passes it an array of the data set used to create predictions, passes it all the possible output values without duplicates
    tree.export_graphviz(model, out_file='music-predictor.dot',
                         feature_names=['age', 'gender'],
                         class_names=sorted(y.unique()),
                         label='all',
                         rounded=True,
                         filled=True)

    #passes the test results we saved for testing along with predictions from our test values to see how accurate our model is
    score = accuracy_score(yTest, predictions)
    print("Accuracy: " + str(score))
    print("Test prediction")

    #give it a test value to predict just for fun
    #passes it the value of a 21 year old male
    testPrediction = model.predict([[21, 1]])

    print("A 21 year old male likes " + testPrediction[0])


start()