import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib  inline
from sklearn import datasets, decomposition, cluster, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage  
import seaborn as sns
from keras.utils import to_categorical

#read data from csv
data = pd.read_csv("dorothea_full.csv")
data_x = pd.DataFrame(data)
data_y = data_x.iloc[:,10001]
#remove first and last columns
data_x1 = data_x.iloc[:,1:10001]

# Turn x and y data into matrices for keras
labels = pd.DataFrame(data_y).as_matrix()
labels = labels.flatten()
labels = to_categorical(labels)

features = pd.DataFrame(data_x1).as_matrix()

from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras import optimizers

# Train / Test data split
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.33, random_state=2)

# Learning rate
learning_rate = 0.01

# Number of training epochs
num_epochs = 100

# Network architecture parameters
num_features = len(train_x[0])
num_classes = len(train_y[0])
num_hidden_nodes = 10

# With sequential we can just add layers to the network
model = Sequential()
model.add(Dense(num_hidden_nodes, input_dim=num_features, activation='relu'))
model.add(Dense(num_hidden_nodes, activation='relu'))
model.add(Dense(num_hidden_nodes, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

# The fit function returns a dictionary of the training and evaluation losses and accuracies
metrics = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=num_epochs, verbose=2)

# Plot the error chart
plt.plot(metrics.history['loss'])
plt.plot(metrics.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Plot the accuracy chart
plt.plot(metrics.history['acc'])
plt.plot(metrics.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

eval_metrics = model.evaluate(test_x, test_y)
print("Evaluation: Loss=" + str(round(eval_metrics[0], 3)) + " Accuracy=" + str(round(eval_metrics[1], 3)))

