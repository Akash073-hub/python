

#Create a 1D NumPy array from a list of numbers from 1 to 10.
import numpy as np
arr = np.arange(10)
print(arr)
     
[0 1 2 3 4 5 6 7 8 9]

#Create a 1D array of 20 equally spaced numbers between 0 and 1.
arr = np.linspace(0, 1, 20)
print(arr)
     
[0.         0.05263158 0.10526316 0.15789474 0.21052632 0.26315789
 0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737
 0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684
 0.94736842 1.        ]

#Generate a 1D array of 10 random integers between 1 and 100.
arr = np.random.randint(1,100,10)
print(arr)
     
[34 49 20 59  6 27 80 84 94 85]

#Create a 2D array with 3 rows and 4 columns filled with zeros.
arr = np.zeros((3,4))
print(arr)
     
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

#Generate a 2D array with the shape (5,5) containing random floating-point numbers.
arr = np.random.rand(5,5)
print(arr)
     
[[0.95338821 0.67988927 0.45599129 0.17312146 0.94581929]
 [0.65754401 0.59966052 0.06075047 0.02697991 0.55803671]
 [0.4334518  0.17873621 0.17092773 0.92990578 0.32174865]
 [0.23630311 0.72414679 0.40480335 0.19482651 0.77328712]
 [0.43191963 0.87073833 0.89151058 0.74123842 0.19498813]]

#Create a 2x2 identity matrix.
arr = np.eye(2)
print(arr)
     
[[1. 0.]
 [0. 1.]]

#Create a 3D array with the shape (3,4,2) filled with zeros.
arr = np.zeros((3,4,2))Untitled0
print(arr)
     
[[[0. 0.]
  [0. 0.]
  [0. 0.]
  [0. 0.]]

 [[0. 0.]
  [0. 0.]
  [0. 0.]
  [0. 0.]]

 [[0. 0.]
  [0. 0.]
  [0. 0.]
  [0. 0.]]]

#Generate a 3D array with the shape (2,3,3) containing random integers between 1 and 10.
arr = np.random.randint(1, 10, (2, 3, 3))
print(arr)
     
[[[1 1 4]
  [3 7 7]
  [4 3 2]]

 [[7 7 9]
  [4 2 1]
  [4 5 4]]]

#Convert a list of 24 sequential numbers into a 3D array with shape (2,3,4).
arr = np.arange(24).reshape(2,3,4)
print(arr)
     
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

#Calculate the mean along the row and column.
mean_row = np.mean(arr, axis=0)
mean_column = np.mean(arr, axis=1)
print("Mean along row : ", mean_row)
print("Mean along column : ", mean_column)
     
Mean along row :  [[4.  2.5 6.5 6. ]
 [6.  2.5 5.5 2. ]
 [6.  3.  3.5 6.5]]
Mean along column :  [[4.66666667 3.33333333 7.         4.        ]
 [6.         2.         3.33333333 5.66666667]]

#Create a 3D array with the shape (2,3,4) filled with ones.
arr = np.ones((2,3,4))
print(arr)
     
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]]

#Generate a 4D array with the shape (2,2,2,2) containing all zeros.
arr = np.zeros((2,2,2,2))
print(arr)
     
[[[[0. 0.]
   [0. 0.]]

  [[0. 0.]
   [0. 0.]]]


 [[[0. 0.]
   [0. 0.]]

  [[0. 0.]
   [0. 0.]]]]

#Given a 1D array [3, 7, 2, 9, 5, 8, 1], extract the elements at indices 1, 3, and 5.
arr = np.array([3, 7, 2, 9, 5, 8, 1])
print(arr[[1,3,5]])
     
[7 9 8]

#From a 2D array with shape (4,5), extract the element at row 2, column 3.
arr = np.random.randint(1, 10, (4,5))
print(arr)
print("Element extracted from row 2, column 3 : ", arr[2,3])
     
[[3 3 2 9 1]
 [3 3 7 6 8]
 [4 6 3 6 1]
 [8 9 3 6 5]]
Element extracted from row 2, column 3 :  6

#From a 2D array, extract the entire second row.
print(arr[1, :])
     
[3 3 7 6 8]

#From a 2D array, extract a subarray containing rows 1 to 3 and columns 2 to 4.
print(arr[1:3, 2:4])
     
[[7 6]
 [3 6]]

#From a 3D array with shape (3,4,5), extract the element at position (depth 1,row 2,col 3).
arr = np.random.randint(1, 10, (3,4,5))
print(arr)
print("Element extracted from position (depth 1,row 2,col 3) : ", arr[1,2,3])
     
[[[3 8 3 5 7]
  [8 1 2 9 4]
  [4 1 1 6 8]
  [4 5 8 8 8]]

 [[2 1 6 3 6]
  [6 7 1 8 7]
  [7 8 3 9 6]
  [7 6 8 5 9]]

 [[1 5 7 1 5]
  [2 8 8 2 6]
  [1 1 9 6 1]
  [4 1 5 9 4]]]
Element extracted from position (depth 1,row 2,col 3) :  9

#From a 3D array, extract the entire first "layer" (all rows and columns of the first depth index).
arr = np.random.randint(1, 10, (3,4,5))
print(arr)
print("Entire first layer : \n", arr[0, :, :])
     
[[[2 1 1 5 3]
  [9 6 1 2 4]
  [9 9 1 9 5]
  [3 2 1 7 6]]

 [[2 7 4 4 5]
  [9 7 5 2 9]
  [5 2 6 7 4]
  [7 3 3 5 6]]

 [[9 4 4 5 5]
  [4 8 9 6 2]
  [3 6 9 3 3]
  [2 7 7 1 5]]]
Entire first layer : 
 [[2 1 1 5 3]
 [9 6 1 2 4]
 [9 9 1 9 5]
 [3 2 1 7 6]]

#From a 3D array of shape (4,5,6), extract a subarray containing depths 1 to 3, rows 2 to 4, and columns 3 to 5.
arr = np.random.randint(1, 10, (4, 5, 6))
print("Original array : ", arr)
print("Extracted array : ", arr[1:4, 2:5, 3:6])
     
Original array :  [[[1 4 1 7 4 7]
  [3 5 7 7 2 3]
  [9 8 5 8 2 5]
  [9 6 9 9 4 5]
  [6 4 1 1 8 6]]

 [[1 4 6 6 4 7]
  [9 8 9 2 1 9]
  [9 2 6 7 2 2]
  [8 7 2 7 2 7]
  [1 3 9 2 3 5]]

 [[9 6 4 2 8 8]
  [5 9 7 7 1 5]
  [8 3 1 7 8 8]
  [4 9 9 9 1 7]
  [8 1 7 3 7 2]]

 [[3 5 8 8 9 9]
  [3 3 4 2 3 3]
  [1 1 8 9 3 5]
  [9 8 1 1 1 8]
  [4 9 3 7 6 1]]]
Extracted array :  [[[7 2 2]
  [7 2 7]
  [2 3 5]]

 [[7 8 8]
  [9 1 7]
  [3 7 2]]

 [[9 3 5]
  [1 1 8]
  [7 6 1]]]

#Reshape a 1D array of 12 elements into a 3x4 2D array
arr = np.arange(12)
print(arr)
print("Reshaped array : \n", arr.reshape(3,4))
     
[ 0  1  2  3  4  5  6  7  8  9 10 11]
Reshaped array : 
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

#Reshape a 2D array of shape (3,4) into a 1D array.
arr = np.random.randint(1, 10, (3,4))
print(arr)
print("Reshaped array : \n", arr.flatten())
     
[[5 9 3 5]
 [6 7 2 5]
 [2 3 7 7]]
Reshaped array : 
 [5 9 3 5 6 7 2 5 2 3 7 7]

#Transpose a 2D array of shape (2,5).
arr = np.random.randint(1, 10, (2, 5))
print(arr)
print("Transposed array : \n", arr.transpose())
     
[[6 4 3 7 2]
 [1 9 6 2 1]]
Transposed array : 
 [[6 1]
 [4 9]
 [3 6]
 [7 2]
 [2 1]]

#Reshape a 3D array of shape (2,3,4) into a 2D array of shape (6,4).
arr = np.random.randint(1, 10, (2,3,4))
print(arr)
print("Reshaped array : \n", arr.reshape(6,4))
     
[[[1 3 6 1]
  [5 2 5 7]
  [8 7 4 1]]

 [[8 5 5 6]
  [4 3 5 1]
  [8 6 5 8]]]
Reshaped array : 
 [[1 3 6 1]
 [5 2 5 7]
 [8 7 4 1]
 [8 5 5 6]
 [4 3 5 1]
 [8 6 5 8]]

#Transpose a 3D array, swapping the first and third dimensions
arr = np.random.randint(1, 10, (2,3,4))
print(arr)
print("Transposed array : \n", arr.transpose((2,0,1)))
     
[[[3 7 3 5]
  [9 1 3 5]
  [9 9 8 9]]

 [[6 7 5 7]
  [4 8 1 6]
  [6 4 7 1]]]
Transposed array : 
 [[[3 9 9]
  [6 4 6]]

 [[7 1 9]
  [7 8 4]]

 [[3 3 8]
  [5 1 7]]

 [[5 5 9]
  [7 6 1]]]

#Stack two 2D arrays of shape (3,4) to create a 3D array of shape (2,3,4).
arr1 = np.random.randint(1, 10, (3, 4))
arr2 = np.random.randint(1, 10, (3, 4))

stacked_arr = np.stack((arr1, arr2), axis=0)

print("Array 1:\n", arr1)
print("\nArray 2:\n", arr2)
print("\nStacked 3D Array:\n", stacked_arr)
print("\nShape of stacked array:", stacked_arr.shape)

     
Array 1:
 [[4 9 1 8]
 [4 8 8 1]
 [3 7 7 1]]

Array 2:
 [[7 8 3 6]
 [1 9 2 1]
 [1 5 6 4]]

Stacked 3D Array:
 [[[4 9 1 8]
  [4 8 8 1]
  [3 7 7 1]]

 [[7 8 3 6]
  [1 9 2 1]
  [1 5 6 4]]]

Shape of stacked array: (2, 3, 4)

#Add two 1D arrays of the same length element-wise.
arr1 = [1, 2, 3, 4, 5]
arr2 = [5, 6, 7, 8, 9]

result = arr1 + arr2
print("Array 1: \n", arr1)
print("Array 2: \n", arr2)
print("Result: \n", result)
     
Array 1: 
 [1, 2, 3, 4, 5]
Array 2: 
 [5, 6, 7, 8, 9]
Result: 
 [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]

#Multiply a 2D array by a scalar value.
arr = np.random.randint(1, 10, (3, 3))
result = arr*2
print("Original array: \n", arr)
print("\nResult: \n", result)
     
Original array: 
 [[4 9 3]
 [6 8 4]
 [3 6 8]]

Result: 
 [[ 8 18  6]
 [12 16  8]
 [ 6 12 16]]

#Calculate the square root of each element in an array.
arr = np.random.randint(1, 10, (3, 3))
result = np.sqrt(arr)
print("Original array: \n", arr)
print("\nResult: \n", result)
     
Original array: 
 [[5 1 3]
 [2 2 7]
 [7 6 1]]

Result: 
 [[2.23606798 1.         1.73205081]
 [1.41421356 1.41421356 2.64575131]
 [2.64575131 2.44948974 1.        ]]

#Perform element-wise addition between two 3D arrays of the same shape.
arr1 = np.random.randint(1, 10, (2, 3, 4))
arr2 = np.random.randint(1, 10, (2, 3, 4))
result = arr1 + arr2
print("Array 1: \n", arr1)
print("\nArray 2: \n", arr2)
print("\nResult: \n", result)
     
Array 1: 
 [[[2 3 4 5]
  [6 4 1 7]
  [8 1 1 9]]

 [[8 3 8 5]
  [1 1 9 9]
  [1 1 5 6]]]

Array 2: 
 [[[6 2 7 4]
  [2 9 5 2]
  [6 8 7 7]]

 [[2 1 3 7]
  [2 2 1 6]
  [5 2 3 5]]]

Result: 
 [[[ 8  5 11  9]
  [ 8 13  6  9]
  [14  9  8 16]]

 [[10  4 11 12]
  [ 3  3 10 15]
  [ 6  3  8 11]]]

#Calculate the mean along the second axis of a 3D array.
arr = np.random.randint(1, 10, (2, 3, 4))
result = np.mean(arr, axis=1)
print("Original array: \n", arr)
print("\nResult: \n", result)
     
Original array: 
 [[[3 8 7 2]
  [4 9 3 9]
  [3 6 8 7]]

 [[8 8 1 8]
  [1 6 2 5]
  [9 3 7 8]]]

Result: 
 [[3.33333333 7.66666667 6.         6.        ]
 [6.         5.66666667 3.33333333 7.        ]]

#Apply a function (like np.sin) to each element of a 3D array.
arr = np.random.randint(1, 10, (2, 3, 4))
result = np.sin(arr)
print("Original array: \n", arr)
print("\nResult: \n", result)
     
Original array: 
 [[[9 8 3 6]
  [4 6 2 9]
  [8 2 1 8]]

 [[9 8 3 9]
  [9 3 2 7]
  [4 7 9 2]]]

Result: 
 [[[ 0.41211849  0.98935825  0.14112001 -0.2794155 ]
  [-0.7568025  -0.2794155   0.90929743  0.41211849]
  [ 0.98935825  0.90929743  0.84147098  0.98935825]]

 [[ 0.41211849  0.98935825  0.14112001  0.41211849]
  [ 0.41211849  0.14112001  0.90929743  0.6569866 ]
  [-0.7568025   0.6569866   0.41211849  0.90929743]]]

#Add a 1D array of shape (3,) to each row of a 2D array with shape (4,3).
arr1d = np.array([1, 2, 3])
arr2d = np.random.randint(1, 10, (4, 3))
result = arr2d + arr1d
print("Array 1D: \n", arr1d)
print("\nArray 2D: \n", arr2d)
print("\nResult: \n", result)
     
Array 1D: 
 [1 2 3]

Array 2D: 
 [[9 9 8]
 [2 5 5]
 [1 7 4]
 [2 5 6]]

Result: 
 [[10 11 11]
 [ 3  7  8]
 [ 2  9  7]
 [ 3  7  9]]

#Multiply a 2D array of shape (3,4) with a 1D array of shape (4,).
arr2d = np.random.randint(1, 10, (3, 4))
arr1d = np.array([2, 4, 6, 8])
result = arr1d * arr2d
print("Array 1D: \n", arr1d)
print("\nArray 2D: \n", arr2d)
print("\nResult: \n", result)
     
Array 1D: 
 [2 4 6 8]

Array 2D: 
 [[5 1 2 7]
 [7 8 7 2]
 [1 4 2 8]]

Result: 
 [[10  4 12 56]
 [14 32 42 16]
 [ 2 16 12 64]]

#Add a 1D array of shape (4,) to each "row" across all "layers" of a 3D array with shape (3,5,4).
arr1d = np.array([1, 2, 3, 4])
arr3d = np.random.randint(1, 10, (3, 5, 4))
result = arr3d + arr1d
print("Array 1D: \n", arr1d)
print("\nArray 3D: \n", arr3d)
print("\nResult: \n", result)
     
Array 1D: 
 [1 2 3 4]

Array 3D: 
 [[[6 4 9 1]
  [7 5 5 1]
  [1 4 2 3]
  [3 7 1 1]
  [9 3 1 5]]

 [[5 1 8 7]
  [5 5 4 2]
  [3 8 6 6]
  [4 3 7 4]
  [6 1 1 3]]

 [[5 3 6 5]
  [1 1 6 3]
  [4 4 3 6]
  [6 9 7 4]
  [6 9 8 4]]]

Result: 
 [[[ 7  6 12  5]
  [ 8  7  8  5]
  [ 2  6  5  7]
  [ 4  9  4  5]
  [10  5  4  9]]

 [[ 6  3 11 11]
  [ 6  7  7  6]
  [ 4 10  9 10]
  [ 5  5 10  8]
  [ 7  3  4  7]]

 [[ 6  5  9  9]
  [ 2  3  9  7]
  [ 5  6  6 10]
  [ 7 11 10  8]
  [ 7 11 11  8]]]

#Multiply a 2D array of shape (3,4) with each "layer" of a 3D array with shape (2,3,4).
arr2d = np.random.randint(1, 10, (3, 4))
arr3d = np.random.randint(1, 10, (2, 3, 4))
result = arr3d * arr2d
print("Array 2d : \n", arr2d)
print("Array 3d : \n", arr3d)
print("Result : \n", result)
     
Array 2d : 
 [[4 3 6 6]
 [7 6 5 4]
 [9 3 7 9]]
Array 3d : 
 [[[1 2 7 7]
  [3 4 6 1]
  [8 8 3 6]]

 [[8 9 1 1]
  [3 6 5 5]
  [1 7 4 9]]]
Result : 
 [[[ 4  6 42 42]
  [21 24 30  4]
  [72 24 21 54]]

 [[32 27  6  6]
  [21 36 25 20]
  [ 9 21 28 81]]]

#Scale each "layer" of a 3D array with shape (4,3,5) by a different scalar value from a 1D array of shape (4,).
arr3D = np.random.randint(1, 10, (4, 3, 5))
scalars = np.array([1, 2, 3, 4])
result = arr3D * scalars[:, np.newaxis, np.newaxis]
print("Array 3D: \n", arr3D)
print("Scalars: \n", scalars)
print("Result: \n", result)
     
Array 3D: 
 [[[5 1 3 3 2]
  [2 7 4 3 1]
  [3 4 6 7 4]]

 [[3 1 3 2 1]
  [9 7 4 8 3]
  [3 6 5 8 2]]

 [[4 8 8 1 8]
  [9 3 1 9 1]
  [8 9 8 5 6]]

 [[6 7 6 2 4]
  [7 4 1 5 6]
  [7 7 4 5 1]]]
Scalars: 
 [1 2 3 4]
Result: 
 [[[ 5  1  3  3  2]
  [ 2  7  4  3  1]
  [ 3  4  6  7  4]]

 [[ 6  2  6  4  2]
  [18 14  8 16  6]
  [ 6 12 10 16  4]]

 [[12 24 24  3 24]
  [27  9  3 27  3]
  [24 27 24 15 18]]

 [[24 28 24  8 16]
  [28 16  4 20 24]
  [28 28 16 20  4]]]

#Create a pandas Series with the values [5, 10, 15, 20, 25].
import pandas as pd
series = pd.Series([5, 10, 15, 20, 25])
print(series)
     
0     5
1    10
2    15
3    20
4    25
dtype: int64

#Create a Series with values [100, 200, 300, 400, 500] and index labels ['a', 'b', 'c'
series = pd.Series([100, 200, 300, 400, 500], index=['a', 'b', 'c', 'd', 'e'])
print(series)
     
a    100
b    200
c    300
d    400
e    500
dtype: int64

#Create a Series from this dictionary: {'apple': 3, 'banana': 5, 'orange': 2}.
data = {'apple': 3, 'banana': 5, 'orange': 2}
series = pd.Series(data)
print(series)
     
apple     3
banana    5
orange    2
dtype: int64

#For the Series s = pd.Series([10, 20, 30, 40, 50]): Find its length Find its mean value Find its maximum value
data = ([10, 20, 30, 40, 50])
series = pd.Series(data)
print("Length : ", len(series))
print("Mean : ", series.mean())
print("Maximum : ", series.max())
     
Length :  5
Mean :  30.0
Maximum :  50

#Create a DataFrame from this dictionary:
data = {
    'Name': ['John', 'Emma', 'Alex'],
    'Age': [25, 30, 22],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print(df)
     
   Name  Age      City
0  John   25  New York
1  Emma   30    London
2  Alex   22     Paris

#Create a simple DataFrame with 3 rows and 2 columns named 'A' and 'B' with any numbers you choose.
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)
print(df)
     
   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9

#Convert this list of lists into a DataFrame with column names 'Product', 'Price', 'Quantity':
data = [
    ['Apple', 1.2, 10],
    ['Banana', 0.5, 15],
    ['Orange', 0.8, 8]
]
df = pd.DataFrame(data, columns=['Product', 'Price', 'Quantity'], index = ['A', 'B', 'C'])
print(df)
     
  Product  Price  Quantity
A   Apple    1.2        10
B  Banana    0.5        15
C  Orange    0.8         8

#For the DataFrame created in question 5:
#Display the column names
#Display the first 2 rows
#Display information about the DataFrame (hint: use .info())
data = {
    'Name': ['John', 'Emma', 'Alex'],
    'Age': [25, 30, 22],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print("Column names : ", df.columns)
print("First 2 rows : \n", df.head(2))
print(df.info())
     
Column names :  Index(['Name', 'Age', 'City'], dtype='object')
First 2 rows : 
    Name  Age      City
0  John   25  New York
1  Emma   30    London
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Name    3 non-null      object
 1   Age     3 non-null      int64 
 2   City    3 non-null      object
dtypes: int64(1), object(2)
memory usage: 204.0+ bytes
None

#Using this DataFrame:
#df = pd.DataFrame({ 'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9] })
#Select column 'A'
#Select columns 'A' and 'C'
#Select the value at row 1, column 'B'

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

col_A = df['A']
print("Column A:\n", col_A)
cols_AC = df[['A', 'C']]
print("\nColumns A and C:\n", cols_AC)
value_B1 = df.at[1, 'B']  # OR df.loc[1, 'B']
print("\nValue at row 1, column 'B':", value_B1)
     
Column A:
 0    1
1    2
2    3
Name: A, dtype: int64

Columns A and C:
    A  C
0  1  7
1  2  8
2  3  9

Value at row 1, column 'B': 5

#question16
df = pd.DataFrame({
    'Name': ['John', 'Emma', 'Alex', 'Sarah', 'Mike'],
    'Age': [25, 30, 22, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Berlin', 'Tokyo'],
    'Salary': [50000, 60000, 45000, 55000, 65000]
})

age_filter = df[df['Age'] > 25]
print("Rows where Age > 25:\n", age_filter)
city_filter = df[df['City'].isin(['London', 'Paris'])]
print("\nRows where City is 'London' or 'Paris':\n", city_filter)
salary_filter = df[df['Salary'].between(45000, 60000)]
print("\nRows where Salary is between 45000 and 60000:\n", salary_filter)
     
Rows where Age > 25:
     Name  Age    City  Salary
1   Emma   30  London   60000
3  Sarah   28  Berlin   55000
4   Mike   32   Tokyo   65000

Rows where City is 'London' or 'Paris':
    Name  Age    City  Salary
1  Emma   30  London   60000
2  Alex   22   Paris   45000

Rows where Salary is between 45000 and 60000:
     Name  Age      City  Salary
0   John   25  New York   50000
1   Emma   30    London   60000
2   Alex   22     Paris   45000
3  Sarah   28    Berlin   55000

#question17
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [9, 10, 11, None]
})

missing_counts = df.isnull().sum()
print("Missing values in each column:\n", missing_counts)

df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:\n", df_dropped)

df_filled = df.fillna(0)
print("\nDataFrame after filling missing values with 0:\n", df_filled)

df['A'].fillna(df['A'].mean(), inplace=True)
print("\nDataFrame after filling missing values in column 'A' with mean:\n", df)

df['B'].fillna(100, inplace=True)
print("\nDataFrame after filling missing values in column 'B' with 100:\n", df)

df.ffill(inplace=True)
print("\nDataFrame after forward filling missing values:\n", df)

     
Missing values in each column:
 A    1
B    1
C    1
dtype: int64

DataFrame after dropping rows with missing values:
      A    B    C
0  1.0  5.0  9.0

DataFrame after filling missing values with 0:
      A    B     C
0  1.0  5.0   9.0
1  2.0  0.0  10.0
2  0.0  7.0  11.0
3  4.0  8.0   0.0

DataFrame after filling missing values in column 'A' with mean:
           A    B     C
0  1.000000  5.0   9.0
1  2.000000  NaN  10.0
2  2.333333  7.0  11.0
3  4.000000  8.0   NaN

DataFrame after filling missing values in column 'B' with 100:
           A      B     C
0  1.000000    5.0   9.0
1  2.000000  100.0  10.0
2  2.333333    7.0  11.0
3  4.000000    8.0   NaN

DataFrame after forward filling missing values:
           A      B     C
0  1.000000    5.0   9.0
1  2.000000  100.0  10.0
2  2.333333    7.0  11.0
3  4.000000    8.0  11.0
<ipython-input-2-a81458f47ea9>:19: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['A'].fillna(df['A'].mean(), inplace=True)
<ipython-input-2-a81458f47ea9>:22: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df['B'].fillna(100, inplace=True)

#question18
import pandas as pd

# Create DataFrame with student test scores
df = pd.DataFrame({
    'Student': ['Alex', 'Bob', 'Charlie', 'David'],
    'Math': [85, 90, 70, 80],
    'Science': [90, 85, 75, 85],
    'English': [80, 95, 80, 75]
})

# Add a new column 'Average' that contains the average score for each student
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

# Sort the DataFrame by the Math scores in descending order
df_sorted = df.sort_values(by='Math', ascending=False)

# Find the student with the highest Science score
highest_science_student = df.loc[df['Science'].idxmax(), 'Student']

# Find all students who scored above 85 in any subject
students_above_85 = df[(df['Math'] > 85) | (df['Science'] > 85) | (df['English'] > 85)]

# Calculate the average score for each subject
subject_avg = df[['Math', 'Science', 'English']].mean()

# Add a new row for a student named 'Eve' with scores Math: 95, Science: 92, English: 88
df.loc[len(df)] = ['Eve', 95, 92, 88, (95 + 92 + 88) / 3]

# Create DataFrame with product sales data
df_products = pd.DataFrame({
    'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Quantity': [10, 5, 15, 8, 12, 7],
    'Price': [100, 200, 100, 150, 200, 100]
})

# Calculate the total quantity for each product
total_quantity = df_products.groupby('Product')['Quantity'].sum()

# Calculate the total revenue for each product
df_products['Revenue'] = df_products['Quantity'] * df_products['Price']
total_revenue = df_products.groupby('Product')['Revenue'].sum()

# Print results
print("\nDataFrame sorted by Math scores:\n", df_sorted)
print("\nStudent with the highest Science score:", highest_science_student)
print("\nStudents who scored above 85 in any subject:\n", students_above_85)
print("\nAverage score for each subject:\n", subject_avg)
print("\nUpdated DataFrame with Eve:\n", df)
print("\nTotal quantity for each product:\n", total_quantity)
print("\nTotal revenue for each product:\n", total_revenue)

     
DataFrame sorted by Math scores:
    Student  Math  Science  English  Average
1      Bob    90       85       95     90.0
0     Alex    85       90       80     85.0
3    David    80       85       75     80.0
2  Charlie    70       75       80     75.0

Student with the highest Science score: Alex

Students who scored above 85 in any subject:
   Student  Math  Science  English  Average
0    Alex    85       90       80     85.0
1     Bob    90       85       95     90.0

Average score for each subject:
 Math       81.25
Science    83.75
English    82.50
dtype: float64

Updated DataFrame with Eve:
    Student  Math  Science  English    Average
0     Alex    85       90       80  85.000000
1      Bob    90       85       95  90.000000
2  Charlie    70       75       80  75.000000
3    David    80       85       75  80.000000
4      Eve    95       92       88  91.666667

Total quantity for each product:
 Product
A    32
B    17
C     8
Name: Quantity, dtype: int64

Total revenue for each product:
 Product
A    3200
B    3400
C    1200
Name: Revenue, dtype: int64

#question19
#Mean Calculation: Given the data set [3, 7, 9, 12, 15], calculate the mean. What does this mean in the context of the data?
import numpy as np
data = [3, 7, 9, 12, 15]
mean = np.mean(data)
print("Mean : ", mean)
#The mean of 9.2 represents the central value of the dataset. In the context of the data, this means that if all values were evenly distributed, each would be approximately 9.2.

#Kurtosis Interpretation: What does kurtosis indicate about a data set? Describe how you would identify whether a data set has high or low kurtosis.
import pandas as pd

data = [3, 7, 9, 12, 15]
kurtosis_value = pd.Series(data).kurt()
print("Kurtosis:", kurtosis_value)
#Kurtosis measures the tailedness of a data distribution, indicating whether the data has heavy or light tails compared to a normal distribution.

#Median vs. Mean: Explain the difference between the mean and median. When is it more appropriate to use the median instead of the mean?
#Mean (Average): The sum of all values divided by the total number of values.
#Median: The middle value in an ordered dataset. If there are an even number of values, it is the average of the two middle numbers.

#Mode Identification: Given the data set [2, 4, 4, 6, 8, 8, 8, 10], identify the mode. How would you interpret the mode in this context?
import pandas as pd

data = [2, 4, 4, 6, 8, 8, 8, 10]
mode_value = pd.Series(data).mode()

print("Mode:", mode_value.tolist())  # Convert to list for better readability

#Range Calculation: Calculate the range for the following data set: [3, 5, 8, 12, 14]. What does the range tell you about the spread of the data?
data = [3, 5, 8, 12, 14]
data_range = max(data) - min(data)
print("Range:", data_range)

#Variance Calculation: Given the data set [1, 3, 5, 7], calculate the sample variance. What does variance tell you about the distribution of the data?
import numpy as np

data = [1, 3, 5, 7]
variance = np.var(data, ddof=1)
print("Sample Variance:", variance)

#Standard Deviation: How does standard deviation help in understanding data spread? Calculate the standard deviation for the data set [5, 7, 10, 12, 14].
import statistics
import numpy as np

data = [5, 7, 10, 12, 14]

pop_std_dev = statistics.pstdev(data)
sample_std_dev = statistics.stdev(data)

numpy_pop_std_dev = np.std(data, ddof=0)
numpy_sample_std_dev = np.std(data, ddof=1)

print("Population Standard Deviation:", pop_std_dev)
print("Sample Standard Deviation:", sample_std_dev)
print("NumPy Population Standard Deviation:", numpy_pop_std_dev)
print("NumPy Sample Standard Deviation:", numpy_sample_std_dev)

#Skewness Analysis: Given the data set [2, 3, 4, 5, 100], describe the skewness of the data and explain what this means.
import scipy.stats as stats

data = [2, 3, 4, 5, 100]
skewness = stats.skew(data)

print("Skewness:", skewness)

     
Mean :  9.2
Kurtosis: -0.5068529725881108
Mode: [8]
Range: 11
Sample Variance: 6.666666666666667
Population Standard Deviation: 3.2619012860600183
Sample Standard Deviation: 3.646916505762094
NumPy Population Standard Deviation: 3.2619012860600183
NumPy Sample Standard Deviation: 3.646916505762094
Skewness: 1.4974854324944105

#question20
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

print(person)

     
{'name': 'John', 'age': 30, 'city': 'New York'}

#question21
# Write a Python program to read the contents of a file named data.txt. Assume the file contains multiple lines of text. Print each line in the file.
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())

     

#question22
# Write a Python program to create a file called output.txt and write the text "Hello, World!" into it. Ensure the file is saved in the current directory.
with open("output.txt", "w") as file:
    file.write("Hello, World!")

     

#question23
# Create a dictionary with the key "product" and the value "laptop". Then, update the dictionary by adding a new key-value pair "price": 1000. Print the updated dictionary.
product_info = {"product": "laptop"}
product_info["price"] = 1000

print(product_info)

     

#question24
# Use the with statement to open a file named example.txt for reading, and then print its contents. What is the advantage of using with for file handling?
with open("example.txt", "r") as file:
    print(file.read())
#advantsges is Ensures automatic file closure, prevents resource leaks, and makes code cleaner and error-free.
     
