##Jose Aguilera
# 11/2019
# A.I. 481
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
class tester:
    """seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("wine.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 1:14]
    Y = dataset[:, 0]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        model = Sequential()
        model.add(Dense(14, input_dim=13, activation='relu'))
        model.add(Dense(10, activation='relu'))
        #model.add(Dense(6, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(X[train], Y[train], epochs=600, batch_size=50, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
        # Load the data
    dataset = loadtxt('wine.csv', delimiter=',')
    # split data
    X = dataset[:, 1:14]
    y = dataset[:, 0]
    # define the keras model
    model = Sequential()
    model.add(Dense(14, input_dim=13, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=400, batch_size=20)
    # evaluate the keras model
    #Testing
    dataset2 = loadtxt('Book.csv', delimiter=',')
    a = dataset2[:, 1:14]
    b = dataset2[:, 0]
    #
    _, accuracy = model.evaluate(a, b)
    print('Accuracy: %.2f' % (accuracy * 100))
    predictions = model.predict_classes(a)
    for i in range(5):
        print('%s => %d (expected %d)' % (a[i].tolist(), predictions[i], b[i]))
        """
    def getModel():
        # Load the data
        dataset = loadtxt('wine.csv', delimiter=',')

        # split data
        X = dataset[:, 1:14]
        y = dataset[:, 0]

        # define the keras model
        model = Sequential()
        model.add(Dense(14, input_dim=13, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        # compile the keras model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X, y, epochs=500, batch_size=20)
        return model


if __name__ == '__main__':
    #tester()
    ok = tester.getModel()