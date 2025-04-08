import numpy as np
array=np.random.randint(1,21, (3,3))
print("\nArray without the mean value : \n", array)
mean = np.mean(array)
print("\nArray with mean value : \n", mean)
array[array < 10 ] = 0
print("\nArray after replacing the value : \n ", array)
