import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class KNearestNeighbours():
    def __init__(self, X_train, Y_train, n_neighbours=5):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_neighbours = n_neighbours

    '''Returns the euclidean distance between a and b'''
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2, axis=1))
    
    '''Returns the 5 shortest distances for all test values'''
    def kneighbours(self, X_test):
        points = []
        index_arr = []
        distance_arr = []
        #Iterate through all test samples
        for x in X_test:
            #Calculate euclidean distance and append to array
            point = self.euclidean_distance(x, self.X_train)
            points.append(point)
        #Iterate through the calculated distances
        for distance in points:
            #Add the index to each distance in the list
            en = enumerate(distance)
            #Save the 5 shortest distances for that test sample
            so = sorted(en, key=lambda x: x[1])[:self.n_neighbours]
            #Separate indexes from enumerate object
            indexes = [i[0] for i in so]
            #Separate distances from enumerate object
            distances = [i[1] for i in so]
            #Save indexes and distances to list
            index_arr.append(indexes)
            distance_arr.append(distances)
        return index_arr
    
    '''Returns the predicted y value of the test sample'''
    def predict(self, X_test):
        #Get the number and indexes of neighbours
        neighbours = self.kneighbours(X_test)
        y_preds = np.array([np.argmax(np.bincount(self.Y_train[neighbour])) for neighbour in neighbours])
        return y_preds