# Python Documentation - Quick Reference

## Basic Syntax

### Variables & Types
```python
# Variable assignment
x = 5                    # int
y = 3.14                # float
name = "Python"         # str
is_valid = True         # bool
items = [1, 2, 3]       # list
coords = (10, 20)       # tuple
unique = {1, 2, 3}      # set
data = {"key": "value"} # dict
```

### Operators
```python
# Arithmetic: +, -, *, /, //, %, **
# Comparison: ==, !=, <, >, <=, >=
# Logical: and, or, not
# Membership: in, not in
```

## Control Flow

### Conditionals
```python
if condition:
    pass
elif other_condition:
    pass
else:
    pass
```

### Loops
```python
# For loop
for item in iterable:
    print(item)

# While loop
while condition:
    pass

# Loop control
break      # Exit loop
continue   # Skip to next iteration
```

### Comprehensions
```python
# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict comprehension
square_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
```

## Functions

### Basic Functions
```python
def function_name(param1, param2, default_param=10):
    """Docstring describing the function"""
    return param1 + param2 + default_param

# Lambda functions
square = lambda x: x**2
```

### Args and Kwargs
```python
def flexible_func(*args, **kwargs):
    # args is a tuple of positional arguments
    # kwargs is a dict of keyword arguments
    pass
```

### Decorators
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # Do something before
        result = func(*args, **kwargs)
        # Do something after
        return result
    return wrapper

@decorator
def my_function():
    pass
```

## Data Structures

### Lists
```python
lst = [1, 2, 3, 4, 5]
lst.append(6)           # Add to end
lst.insert(0, 0)        # Insert at index
lst.remove(3)           # Remove first occurrence
lst.pop()               # Remove and return last item
lst.extend([7, 8])      # Add multiple items
lst.sort()              # Sort in place
sorted_lst = sorted(lst) # Return sorted copy
```

### Dictionaries
```python
d = {"a": 1, "b": 2}
d["c"] = 3              # Add/update
d.get("a", default=0)   # Safe get with default
d.keys()                # Get all keys
d.values()              # Get all values
d.items()               # Get (key, value) pairs
d.pop("a")              # Remove and return value
```

### Sets
```python
s = {1, 2, 3}
s.add(4)                # Add element
s.remove(2)             # Remove (error if not found)
s.discard(2)            # Remove (no error)
s1.union(s2)            # s1 | s2
s1.intersection(s2)     # s1 & s2
s1.difference(s2)       # s1 - s2
```

### Strings
```python
s = "Hello World"
s.lower()               # Convert to lowercase
s.upper()               # Convert to uppercase
s.strip()               # Remove whitespace
s.split()               # Split into list
s.replace("Hello", "Hi") # Replace substring
s.startswith("Hello")   # Check prefix
s.endswith("World")     # Check suffix
f"{variable}"           # f-string formatting
```

## Classes

### Basic Class
```python
class MyClass:
    class_variable = "shared"
    
    def __init__(self, value):
        self.value = value  # Instance variable
    
    def method(self):
        return self.value
    
    @classmethod
    def class_method(cls):
        return cls.class_variable
    
    @staticmethod
    def static_method():
        return "static"
    
    def __str__(self):
        return f"MyClass({self.value})"
```

### Inheritance
```python
class Parent:
    def __init__(self):
        self.parent_attr = "parent"

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.child_attr = "child"
```

## File I/O

```python
# Reading
with open("file.txt", "r") as f:
    content = f.read()          # Read entire file
    lines = f.readlines()       # Read all lines

# Writing
with open("file.txt", "w") as f:
    f.write("text")
    f.writelines(["line1\n", "line2\n"])

# Append
with open("file.txt", "a") as f:
    f.write("more text")
```

## Exception Handling

```python
try:
    risky_operation()
except SpecificError as e:
    handle_error(e)
except (Error1, Error2):
    handle_multiple()
except Exception as e:
    handle_generic(e)
else:
    # Runs if no exception
    pass
finally:
    # Always runs
    cleanup()
```

## Common Built-in Functions

```python
len(iterable)           # Length
max(iterable)           # Maximum value
min(iterable)           # Minimum value
sum(iterable)           # Sum of elements
range(start, stop, step) # Generate sequence
enumerate(iterable)     # Index, value pairs
zip(iter1, iter2)       # Combine iterables
map(func, iterable)     # Apply function
filter(func, iterable)  # Filter elements
any(iterable)           # True if any element is true
all(iterable)           # True if all elements are true
```

## Modules & Imports

```python
import module
from module import function
from module import *
import module as alias
from package.module import Class
```

## Context Managers

```python
class MyContext:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code
        return False  # Propagate exceptions

with MyContext() as ctx:
    pass
```

## Common Standard Library

```python
import os              # OS operations
import sys             # System operations
import math            # Math functions
import random          # Random numbers
import datetime        # Date and time
import json            # JSON handling
import re              # Regular expressions
import collections     # Specialized containers
import itertools       # Iterator functions
import functools       # Higher-order functions
```
