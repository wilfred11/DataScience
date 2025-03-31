import numpy as np
import pandas as pd
from numpy import mean
from sklearn.preprocessing import StandardScaler

#https://www.youtube.com/watch?v=MQzG1hfhow4

def mse(actual, predicted):
    return np.mean((actual-predicted)**2)

def mse_grad(actual, predicted):
    return (predicted - actual)

def init_layers(inputs):
    layers = []
    for i in range(1, len(inputs)):
        layers.append([
            np.random.rand(inputs[i-1], inputs[i]) / 5 - .1, #weights between -1 and +1
            np.ones((1,inputs[i])) #biases to 1
        ])
    return layers

def forward(batch, layers):
    hidden = [batch.copy()]
    for i in range(len(layers)):
        batch = np.matmul(batch, layers[i][0]) + layers[i][1]
        if i < len(layers) - 1:
            batch = np.maximum(batch, 0)
        hidden.append(batch.copy())

    return batch, hidden

def backward(layers, hidden, grad, lr):
    for i in range(len(layers)-1, -1, -1):
        if i != len(layers) - 1:
            grad = np.multiply(grad, np.heaviside(hidden[i+1], 0))

        grad = grad.T
        w_grad = np.matmul(grad, hidden[i]).T
        b_grad = np.mean(grad.T, axis=0)

        layers[i][0] -= (w_grad + layers[i][0] * .01) * lr
        layers[i][1] -= b_grad * lr
        grad = np.matmul(layers[i][0], grad).T
    return layers

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

def neural_net():
    # Read in the data
    data = pd.read_csv("Data/clean_weather.csv", index_col=0)
    # Fill in any missing values in the data with past values
    data = data.ffill()

    # Create a scatter plot of tmax and tmax_tomorrow
    data.plot.scatter("tmax", "tmax_tomorrow")
    PREDICTORS = ["tmax", "tmin", "rain"]
    TARGET = "tmax_tomorrow"

    # Scale our data so relu works better
    # All temperature values in the original dataset are over 0, so relu won't do much for several epochs
    # Scaling will make some of the input data negative
    scaler = StandardScaler()
    data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

    split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d
                                                                in
                                                                split_data]

    layer_conf = [3, 10, 10, 1]
    lr = 1e-6
    epochs = 10
    batch_size = 8

    layers = init_layers(layer_conf)
    print(len(layers[1]))

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, train_x.shape[0], batch_size):
            x_batch = train_x[i:(i + batch_size)]
            y_batch = train_y[i:(i + batch_size)]
            pred, hidden = forward(x_batch, layers)

            loss = mse_grad(y_batch, pred)
            epoch_loss.append(np.mean(loss ** 2))

            layers = backward(layers, hidden, loss, lr)

        valid_preds, _ = forward(valid_x, layers)

        print(f"Epoch: {epoch} Train MSE: {mean(epoch_loss)} Valid MSE: {mse(valid_preds, valid_y)}")