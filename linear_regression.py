import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def transposed_matrix(a):
    return a.transpose()
def multiplication_of_matrices(b,c):
    return np.dot(b,c)
def inverse_matrix(d):
    return 1/d

# Load the dataset from the Income1.csv file
data_income1 = pd.read_csv('Income1.csv')
x_train = data_income1['Education'].values # Axis x_train (Education)
y_train = data_income1['Income'].values # Axis y_train (Income)


m_x = np.array(x_train) # Matrix in x_train
m_y = np.array(y_train) # Matrix in y_train
m_x_t = transposed_matrix(m_x) # Matrix transposed in x_train

w1 = multiplication_of_matrices(m_x_t, m_x) # Call the matrix multiplication function
w1_inv = inverse_matrix(w1) # Call the matrix inversion function 
w2 = multiplication_of_matrices(w1_inv, m_x_t) # Call the matrix multiplication function
w = multiplication_of_matrices(w2, m_y) # Call the matrix multiplication function
print(w)
# Calculation of y predictive
y_pred = x_train*w

# Plot of Data(x_train, y_train) and Linear Regression 
plt.scatter(x_train, y_train, color='blue', label='Data (x_train, y_train)')
plt.plot(x_train, y_pred, color='red', label='Linear Regression')
plt.xlabel('Education')
plt.ylabel('Income')
plt.legend()
plt.grid()
plt.show()

# Calculation of MSE

w_values = np.linspace(0, 5, 30) # Auxiliary variable
mse_values = [] # Initializes the vector with zero

for w_test in w_values:
    y_pred_temp = w_test*x_train # Calculation of y temporary predictive
    mse = np.mean((y_train - y_pred_temp)**2) # Calculation of mean square error samples=30
    mse_values.append(mse) # Stores the data in the mse_values array

#Plot of MSE curve
mse_w = np.mean((y_train - y_pred)**2)

plt.plot(w_values, mse_values, color='green')
plt.scatter(w, mse_w, color='black')

plt.xlabel('Value of w')
plt.ylabel('Mean Square Error (MSE)')
plt.grid()
plt.show()