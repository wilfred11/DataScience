import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# calculates the predicted values based on the input, weight, and bias.
def linear_regression_equation(input_value, weight, bias):
    # X = independent value
    # Y = Dependent Value
    # M = SLOPE
    # B = INTERCEPT/BIAS
    # = y=mx+b
    predicted_value = (weight * input_value) + bias
    return predicted_value
# calculates the mean squared error (MSE)
def cost_function(input_value, weight, bias, target_value):
    predicted_value = linear_regression_equation(input_value, weight, bias)
    difference = (target_value - predicted_value)**2
    return sum(difference)/len(difference)

# calculates gradients using finite difference approximation.
def gradient_value_using_approx(inputX, outputY, weight, bias):
    # this approach is easy to implement but,
    # it takes more computation power and time.
    f = cost_function
    h = 0.001
    w_grad_val = (f(inputX, weight+h, bias, outputY)-f(inputX, weight, bias, outputY))/h
    b_grad_val = (f(inputX, weight, bias+h, outputY)-f(inputX, weight, bias, outputY))/h
    return (w_grad_val, b_grad_val)

#.... Same Code as given in previous example ....
def gradient_value_using_rules(inputX, outputY, weight, bias):
    # recommended way
    # using chain rule to get derivate of
    # cost_function(linear_regression_equation(input))
    #
    w_grad_val = sum((-2 *inputX)*(outputY - ((weight*inputX)+bias)))/len(inputX)
    b_grad_val = sum(-2*(outputY - ((weight*inputX)+bias)))/len(inputX)
    return (w_grad_val, b_grad_val)

def gradient_descent():
    # an array ranging from -0.5 to 0.5 with a step of 0.01
    inputX = np.arange(-0.5, 0.5, 0.01)
    # random values with a normal distribution to add noise to the input.
    noise = np.random.normal(0, 0.2, inputX.shape)
    # output values with noise.
    outputY = inputX + noise
    print("Input Values : ", inputX[:2], " Output Values : ", outputY[:2])
    # so input dataset is ready inputX, outputY
    plt.scatter(inputX, outputY, c="blue", label="Dataset")

    # initial weights and bias
    weight = 0.1  # any value
    bias = 1  # any value
    # the loss (MSE) before any learning

    before_loss_value = cost_function(inputX, weight, bias, outputY)
    plt.plot(inputX, linear_regression_equation(inputX, weight, bias), c="orange", label="Before Learning")

    # training parameters
    epochs = 300
    learning_rate = 0.08
    # Weights and bias are updated using the learning rate
    # and gradients to minimize the loss.
    for _ in range(epochs):
        #(w_grad_val, b_grad_val) = gradient_value_using_approx(inputX, outputY, weight, bias)
        (w_grad_val, b_grad_val) = gradient_value_using_rules(inputX, outputY, weight, bias)
        weight = weight - (learning_rate * w_grad_val)
        bias = bias - (learning_rate * b_grad_val)

    print("weight after: "+str(weight))
    print("bias after: "+str(bias))

    (slp, inter, rvalue, pvalue, stderr) = linregress(inputX, outputY)
    print("slp: "+str(slp))
    print("intercept: " + str(inter))
    y_pred = inter + slp * inputX
    plt.plot(inputX, y_pred, color="purple", label="Fitted line")

    # the loss (MSE) after the specified number of epochs.
    after_loss_value = cost_function(inputX, weight, bias, outputY)
    print(f"Loss Value (Before Learning) : {before_loss_value}, Loss Value (After Learning) :  {after_loss_value}")
    # plot the linear regression line in green
    plt.plot(inputX, linear_regression_equation(inputX, weight, bias), c="green",
             label=f"After {epochs} epochs learning")
    plt.legend()
    plt.grid(True)
    plt.show()