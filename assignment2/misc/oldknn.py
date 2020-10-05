
class KNNClassifier:
    """
    class KNNClassifier
    run k nearest neighbors classification
    """

    def __init__(self, num_neighbors=5):
        """
        __init__
        set num_neighbors, and create training_data and training_targets objects
        """
        self.num_neighbors = num_neighbors
        self.training_data = None
        self.training_targets = None

    def _get_neighbor_classes(self, test_row):
        """
        _get_neighbor_classes
        gets the classes from k neighbors, by first getting all the distances,
        sorting them, and returning the classes associated with the k closest
        neighbors
        """
        distances = []
        for i in range(len(self.training_data)):
            dist = _get_distance_cosine(self.training_data[i], test_row)
            distances.append([dist, self.training_targets[i]])
        distances = np.array(distances)
        # print(distances)
        distances = distances[distances[:, 0].argsort()]
        # print(distances)
        neighbor_classes = []
        for i in range(self.num_neighbors):
            neighbor_classes.append(distances[i][1])
        return np.array(neighbor_classes)

    def _get_prediction(self, test_row):
        """
        _get_prediction
        gets the prediction for a given row. does this by getting
        all of the neighbor's classes, and finding the one with the
        most hits
        """
        neighbor_classes = self._get_neighbor_classes(test_row)
        classification_count = {}
        max_class = neighbor_classes[0]
        max_class_count = 0
        # print(neighbor_classes)
        for class_name in neighbor_classes:
            if class_name not in classification_count:
                classification_count[class_name] = 1
            else:
                classification_count[class_name] += 1
            if classification_count[class_name] > max_class_count:
                max_class = class_name
                max_class_count = classification_count[class_name]
        # print(classification_count)
        # print(max_class, max_class_count)
        return max_class

    def fit(self, training_data, training_targets):
        """
        fit
        fit is supposed to generate the model, but since this is dependent
        on the input data (and there are no optimizations), the only thing
        it is doing is saving the training data
        """
        self.training_data = training_data
        self.training_targets = training_targets

    def predict(self, testing_data, num_neighbors=None):
        """
        predict
        runs the knn algorithm to find the closest neighbors and get the overall
        classification based off of them
        """
        # if there is no training data you have a problem
        if self.training_data is None or self.training_targets is None:
            raise ValueError("cannot find training data")
        if num_neighbors is not None:
            self.num_neighbors = num_neighbors
        predictions = np.array([])
        for test_row in testing_data:
            predictions = np.append(
                predictions, self._get_prediction(test_row))
        # print(predictions)
        return predictions