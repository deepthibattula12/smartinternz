import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-10,10,400)
y=x**2
plt.figure(figsize=(8,6))
plt.plot(x,y,label="y=x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("plot the function y=x^2")
plt.legend()
plt.grid(True)
plt.show()
