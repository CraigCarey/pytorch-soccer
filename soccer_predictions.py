#!/usr/bin/env python

# medium.com/@andreluiz_4916/pytorch-neural-networks-to-predict-matches-results-in-soccer-championships-part-i-a6d0eefeca51

import pandas
import torch
import matplotlib.pyplot as plt
import numpy as np

filepath = "data/training_2010.csv"
df = pandas.read_csv(filepath)


# Extract the features we're interested in
extract = [5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 47, 48, 49]
df = df.iloc[:, extract]

# Train/test split
training = df.iloc[:-20] # select all the rows except for the last 20
test = df.iloc[-20:] # select the last 20 rows

# Normalize the data between 0 and 1
for e in range(len(training.columns) - 3): # not normalising the last 3 (target) columns
    num = max(training.iloc[:, e].max(), test.iloc[:, e].max()) #check the maximum value in each column
    if num < 10:
        training.iloc[:, e] /= 10
        test.iloc[:, e] /= 10
    elif num < 100:
        training.iloc[:, e] /= 100
        test.iloc[:, e] /= 100
    elif num < 1000:
        training.iloc[:, e] /= 1000
        test.iloc[:, e] /= 1000
    else:
        print("Error in normalization! Please check!")


training = training.sample(frac=1) #shuffle the training data
test = test.sample(frac=1) #shuffle the test data

# all rows, all columns except for the last 3 columns
training_input  = training.iloc[:, :-3]

# all rows, the last 3 columns
training_output = training.iloc[:, -3:]

# all rows, all columns except for the last 3 columns
test_input  = test.iloc[:, :-3]

# all rows, the last 3 columns
test_output = test.iloc[:, -3:]

def convert_output_win(source):
    target = source.copy() # make a copy from source
    target['new'] = 2 # create a new column with any value
    for i, rows in target.iterrows():
        if rows['win'] == 1:
            rows['new'] = 1
        if rows['draw'] == 1:
            rows['new'] = 0
        if rows['defeat'] == 1:
            rows['new'] = 0
    return target.iloc[:, -1]  # return all rows, the last column

training_output = convert_output_win(training_output)
test_output = convert_output_win(test_output)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid() 
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

# convert to tensors
training_input = torch.FloatTensor(training_input.values)
training_output = torch.FloatTensor(training_output.values)
test_input = torch.FloatTensor(test_input.values)
test_output = torch.FloatTensor(test_output.values)

input_size = training_input.size()[1] # number of features selected
hidden_size = 30 # number of nodes/neurons in the hidden layer
model = Net(input_size, hidden_size) # create the model
criterion = torch.nn.BCELoss() # works for binary classification

# without momentum parameter
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.9)

# with momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)

model.eval()
y_pred = model(test_input)
before_train = criterion(y_pred.squeeze(), test_output)
print('Test loss before training' , before_train.item())

model.train()
epochs = 5000
errors = []

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(training_input)

    # Compute Loss
    loss = criterion(y_pred.squeeze(), training_output)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

    # Backward pass
    loss.backward()
    optimizer.step()

model.eval()
y_pred = model(test_input)
after_train = criterion(y_pred.squeeze(), test_output)
print('Test loss after Training' , after_train.item())

def plotcharts(errors):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(test_output.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=7)
    plt.show()

plotcharts(errors)
