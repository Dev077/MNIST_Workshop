# NumPy Cheatsheet

## Import
```python
import numpy as np
```

## Array Creation

```python
# From lists
np.array([1, 2, 3])
np.array([[1, 2], [3, 4]])

# Initialization
np.zeros((3, 4))                # All zeros
np.ones((3, 4))                 # All ones
np.empty((3, 4))                # Uninitialized
np.full((3, 4), 7)              # Fill with value
np.eye(3)                       # Identity matrix
np.diag([1, 2, 3])             # Diagonal matrix

# Range arrays
np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)           # 5 evenly spaced [0, 1]
np.logspace(0, 2, 5)           # 5 log-spaced [10^0, 10^2]

# Random
np.random.rand(3, 4)           # Uniform [0, 1)
np.random.randn(3, 4)          # Standard normal
np.random.randint(0, 10, (3, 4)) # Random integers
np.random.choice([1, 2, 3], 5) # Random choice
```

## Array Attributes

```python
arr.shape                      # Dimensions
arr.size                       # Total elements
arr.ndim                       # Number of dimensions
arr.dtype                      # Data type
arr.itemsize                   # Size of each element in bytes
```

## Indexing & Slicing

```python
arr[0]                         # First element/row
arr[-1]                        # Last element/row
arr[1:4]                       # Slice [1, 4)
arr[::2]                       # Every 2nd element
arr[::-1]                      # Reverse

# 2D
arr[0, 1]                      # Row 0, Column 1
arr[0:2, 1:3]                  # Rows 0-1, Cols 1-2
arr[:, 1]                      # All rows, Col 1
arr[0, :]                      # Row 0, all cols

# Boolean indexing
arr[arr > 5]                   # Elements > 5
arr[(arr > 5) & (arr < 10)]    # Compound condition

# Fancy indexing
arr[[0, 2, 4]]                # Select rows 0, 2, 4
arr[[0, 1], [1, 2]]           # Elements at (0,1) and (1,2)
```

## Reshaping

```python
arr.reshape(3, 4)              # Reshape to 3x4
arr.ravel()                    # Flatten to 1D
arr.flatten()                  # Flatten (copy)
arr.T                          # Transpose
arr.transpose()                # Transpose
arr.squeeze()                  # Remove single dimensions
arr[np.newaxis, :]             # Add dimension
```

## Concatenation & Splitting

```python
np.concatenate([arr1, arr2])   # Join arrays
np.vstack([arr1, arr2])        # Stack vertically
np.hstack([arr1, arr2])        # Stack horizontally
np.stack([arr1, arr2])         # Stack along new axis
np.split(arr, 3)               # Split into 3 parts
np.vsplit(arr, 2)              # Vertical split
np.hsplit(arr, 2)              # Horizontal split
```

## Mathematical Operations

```python
# Element-wise
arr + 5, arr - 5, arr * 5, arr / 5
arr + arr2, arr * arr2         # Element-wise ops
arr ** 2                       # Power
np.sqrt(arr)                   # Square root
np.exp(arr)                    # Exponential
np.log(arr)                    # Natural log
np.log10(arr)                  # Base 10 log
np.abs(arr)                    # Absolute value

# Trigonometric
np.sin(arr), np.cos(arr), np.tan(arr)
np.arcsin(arr), np.arccos(arr), np.arctan(arr)

# Rounding
np.round(arr, 2)               # Round to 2 decimals
np.floor(arr)                  # Round down
np.ceil(arr)                   # Round up

# Matrix operations
np.dot(arr1, arr2)             # Dot product
arr1 @ arr2                    # Matrix multiplication
np.linalg.inv(arr)             # Inverse
np.linalg.det(arr)             # Determinant
np.linalg.eig(arr)             # Eigenvalues/vectors
```

## Aggregation Functions

```python
arr.sum()                      # Sum all elements
arr.sum(axis=0)                # Sum along axis 0
arr.mean()                     # Mean
arr.std()                      # Standard deviation
arr.var()                      # Variance
arr.min()                      # Minimum
arr.max()                      # Maximum
arr.argmin()                   # Index of minimum
arr.argmax()                   # Index of maximum
arr.cumsum()                   # Cumulative sum
arr.cumprod()                  # Cumulative product

# Percentiles
np.median(arr)
np.percentile(arr, 25)         # 25th percentile
np.quantile(arr, 0.75)         # 75th quantile
```

## Broadcasting

```python
# Arrays with different shapes can be operated together
arr1 = np.array([[1], [2], [3]])  # Shape (3, 1)
arr2 = np.array([10, 20, 30])     # Shape (3,)
result = arr1 + arr2                # Shape (3, 3)
```

## Comparison & Logic

```python
arr > 5                        # Boolean array
arr == 5
np.all(arr > 0)               # True if all > 0
np.any(arr > 0)               # True if any > 0
np.logical_and(arr1, arr2)
np.logical_or(arr1, arr2)
np.logical_not(arr)
np.where(arr > 5, 1, 0)       # Conditional replacement
```

## Sorting & Searching

```python
np.sort(arr)                   # Return sorted copy
arr.sort()                     # Sort in-place
np.argsort(arr)               # Indices that would sort
np.argwhere(arr > 5)          # Indices where condition true
np.unique(arr)                # Unique elements
np.unique(arr, return_counts=True)  # Unique with counts
```

## Set Operations

```python
np.intersect1d(arr1, arr2)    # Common elements
np.union1d(arr1, arr2)        # Union
np.setdiff1d(arr1, arr2)      # In arr1 but not arr2
np.in1d(arr1, arr2)           # Boolean for arr1 elements in arr2
```

## Statistics

```python
np.mean(arr)
np.median(arr)
np.std(arr)                    # Standard deviation
np.var(arr)                    # Variance
np.corrcoef(arr1, arr2)       # Correlation coefficient
np.cov(arr1, arr2)            # Covariance
np.histogram(arr, bins=10)    # Histogram
```

## Linear Algebra (np.linalg)

```python
np.linalg.norm(arr)           # Vector/matrix norm
np.linalg.inv(arr)            # Matrix inverse
np.linalg.det(arr)            # Determinant
np.linalg.eig(arr)            # Eigenvalues and eigenvectors
np.linalg.svd(arr)            # Singular value decomposition
np.linalg.solve(A, b)         # Solve Ax = b
np.linalg.lstsq(A, b)         # Least squares solution
```

## File I/O

```python
np.save('array.npy', arr)      # Save single array
np.load('array.npy')           # Load array
np.savez('arrays.npz', a=arr1, b=arr2)  # Save multiple
data = np.load('arrays.npz')
arr1 = data['a']

# Text files
np.savetxt('data.txt', arr)
np.loadtxt('data.txt')
np.genfromtxt('data.csv', delimiter=',')
```

## Useful Functions

```python
np.clip(arr, min, max)        # Clip values
np.pad(arr, pad_width=1)      # Pad array
np.roll(arr, shift=2)         # Roll elements
np.flip(arr)                  # Reverse array
np.tile(arr, (2, 3))          # Repeat array
np.repeat(arr, 3)             # Repeat elements
np.meshgrid(x, y)             # Create coordinate matrices
np.nan_to_num(arr)            # Replace NaN with zero
```

## Copy vs View

```python
arr2 = arr                    # Reference (same object)
arr2 = arr.view()             # Shallow copy (shares data)
arr2 = arr.copy()             # Deep copy (independent)
```
