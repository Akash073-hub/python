import matplotlib.pyplot as plt
import numpy as np
data = np.random.randint(1, 9, 1000)
print(data)
plt.hist(data, bins=30,
color='purple', alpha=0.7)
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.show()
