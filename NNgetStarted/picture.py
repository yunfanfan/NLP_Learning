import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def f_1(x, A, B):
    return A * x + B

input_vecs = [5, 3, 8, 1.4, 10.1]
labels = [5500, 2300, 7600, 1800, 11400]
plt.scatter(input_vecs[:], labels[:], 25, "red")

A1, B1 = optimize.curve_fit(f_1, input_vecs, labels)[0]
x1 = np.arange(0, 14, 1)
y1 = A1 * x1 + B1
plt.plot(x1, y1, "blue")

plt.title("curve_fit")
plt.xlabel('year')
plt.ylabel('salary')

plt.show()