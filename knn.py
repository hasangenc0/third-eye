import csv
import random
import math
import operator
from decimal import Decimal

class KNClassifier:

    def __init__(self):
        self.dataSet = []
        self.trainingSet = []
        self.testSet = []
        self.predictions = []
        self.label = 0

    def load_dataset(self, filename, label, batch=100, titles=False):
        with open(filename, 'rt') as csv_file:
            lines = csv.reader(csv_file)
            dataset = list(lines)
            if titles:
                del dataset[0]

            length = int(len(dataset)*batch/100)

            if length < 1:
                raise Exception('Data size after batching is too small')

            self.dataSet = [[] for i in range(length)]
            self.label = label
            for c, data in enumerate(dataset):
                if c >= length:
                    break
                for i, value in enumerate(data):
                    if i != self.label-1:
                        self.dataSet[c].append(self.get_value_for_data(value))
                    else:
                        self.dataSet[c].append(self.get_value_for_data(value))

    def split_data(self, proportion=0.5):
        for _, data in enumerate(self.dataSet):
            if random.random() < proportion:
                self.trainingSet.append(data)
            else:
                self.testSet.append(data)

    def load_data_for_predict(self, filename, titles=False):
        with open(filename, 'rt') as csv_file:
            lines = csv.reader(csv_file)
            dataset = list(lines)
            if titles:
                del dataset[0]

            self.trainingSet = self.dataSet
            self.testSet = [[] for i in range(len(dataset))]
            for c, data in enumerate(dataset):
                for i, value in enumerate(data):
                    if i != self.label-1:
                        self.testSet[c].append(self.get_value_for_data(value))
                    else:
                        self.testSet[c].append(value)

    @staticmethod
    def get_value_for_data(value):
        try:
            return float(value)
        except Exception:
            return KNClassifier.string_to_number(value)

    @staticmethod
    def string_to_number(s):
        return int.from_bytes(s.encode(), 'little')

    @staticmethod
    def number_to_string(n):
        return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

    @staticmethod
    def euclidean_distance(test_instance, training_instance, label):
        distance = 0
        for i, test in enumerate(test_instance):
            if i != label-1:
                distance += Decimal(pow((test - training_instance[i]), 2))
        return math.sqrt(distance)

    def get_neighbors(self, test_instance, training_set, k):
        distances = []
        for _, data in enumerate(training_set):
            dist = self.euclidean_distance(test_instance, data, self.label)
            distances.append((data, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    @staticmethod
    def get_response(neighbors, label):
        sum = 0
        for _, neighbor in enumerate(neighbors):
            sum += neighbor[label]

        return sum/len(neighbors)

    def get_accuracy(self):
        correct = 0
        for i in range(len(self.testSet)):
            if self.testSet[i][-1] == self.predictions[i][0]:
                correct += 1
        return (correct / float(len(self.testSet))) * 100.0

    def predict(self, k):
        self.predictions = []
        for i, data in enumerate(self.testSet):
            neighbors = self.get_neighbors(data, self.trainingSet, k)
            result = self.get_response(neighbors, self.label-1)
            self.predictions.append([result, KNClassifier.get_value_for_data(data[self.label-1])])
        return self.predictions
