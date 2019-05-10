import csv
import numpy as np


#Return training, test, and validation data from the fashion mnist csv files.
def load_data():
    training_data, validation_data, test_data = [], [], []
    
    with open('fashion-mnist_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0: #column headers
                line_count += 1
            elif line_count <= 5000:
                # (np array of pixel values, unit vector label of image)
                unit_vec = np.zeros((10,1))
                unit_vec[int(row[0])] = 1.0
                training_data.append((np.asarray([int(x) for x in row[1:]]).reshape(784,1),unit_vec))
                line_count += 1
            elif line_count <= 6000:
                # (np array of pixel values, int label of image)
                validation_data.append((np.asarray([int(x) for x in row[1:]]).reshape(784,1),int(row[0])))
                line_count += 1
            else:
                break
    
    with open('fashion-mnist_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0: #column headers
                line_count += 1
            elif line_count <= 1000:
                # (np array of pixel values, int label of image)
                test_data.append((np.asarray([int(x) for x in row[1:]]).reshape(784,1),int(row[0])))
                line_count += 1
            else:
                break            
            
    return (training_data, validation_data, test_data)


if "__main__" == __name__:
    load_data()