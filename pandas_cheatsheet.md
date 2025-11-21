# Pandas Cheatsheet

## Import
```python
import pandas as pd
import numpy as np
```

## Creating Data Structures

### Series
```python
s = pd.Series([1, 2, 3, 4])
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
```

### DataFrame
```python
# From dictionary
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': ['a', 'b', 'c']
})

# From list of lists
df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])

# From numpy array
df = pd.DataFrame(np.random.randn(4, 3), columns=['A', 'B', 'C'])

# From CSV
df = pd.read_csv('file.csv')
```

## Reading & Writing Data

```python
# CSV
df = pd.read_csv('file.csv')
df.to_csv('output.csv', index=False)

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', index=False)

# JSON
df = pd.read_json('file.json')
df.to_json('output.json')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)
df.to_sql('table_name', connection)

# Clipboard
df = pd.read_clipboard()
df.to_clipboard()
```

## Viewing Data

```python
df.head(5)                     # First 5 rows
df.tail(5)                     # Last 5 rows
df.sample(5)                   # Random 5 rows
df.info()                      # Summary info
df.describe()                  # Statistical summary
df.shape                       # (rows, columns)
df.columns                     # Column names
df.index                       # Row indices
df.dtypes                      # Column data types
df.values                      # Numpy array of values
df.memory_usage()              # Memory usage
```

## Selection & Indexing

### Columns
```python
df['col']                      # Single column (Series)
df[['col1', 'col2']]          # Multiple columns (DataFrame)
df.col                         # Access column as attribute
```

### Rows
```python
df[0:3]                        # Slice rows 0-2
df.iloc[0]                     # Row by integer position
df.iloc[0:3]                   # Rows 0-2
df.iloc[[0, 2, 4]]            # Specific rows
df.loc['row_label']            # Row by label
df.loc[df['col'] > 5]         # Boolean indexing
```

### Cells
```python
df.iloc[0, 1]                  # Value at row 0, col 1
df.loc['row', 'col']          # By labels
df.at['row', 'col']           # Fast scalar access
df.iat[0, 1]                  # Fast integer access
```

### Boolean Indexing
```python
df[df['col'] > 5]             # Filter rows
df[(df['A'] > 5) & (df['B'] < 10)]  # Multiple conditions
df[df['col'].isin([1, 2, 3])] # Values in list
df[~df['col'].isin([1, 2])]   # Not in list
```

## Adding & Removing

### Columns
```python
df['new_col'] = values         # Add column
df['sum'] = df['A'] + df['B'] # Computed column
df.insert(1, 'new', values)   # Insert at position
df.drop('col', axis=1)        # Remove column
df.drop(['A', 'B'], axis=1)   # Remove multiple
del df['col']                  # Delete column
```

### Rows
```python
df.append(new_row, ignore_index=True)  # Add row
df = pd.concat([df, new_df])  # Append DataFrame
df.drop(0, axis=0)            # Remove row by index
df.drop([0, 1, 2])            # Remove multiple rows
```

## Modifying Data

```python
df['col'] = df['col'] * 2     # Modify column
df.loc[0, 'col'] = 10         # Modify cell
df.rename(columns={'old': 'new'})  # Rename columns
df.rename(index={0: 'first'}) # Rename index
df['col'].replace(0, np.nan)  # Replace values
df.fillna(0)                  # Fill NaN with 0
df.dropna()                   # Drop rows with NaN
df.drop_duplicates()          # Remove duplicates
```

## Sorting

```python
df.sort_values('col')         # Sort by column
df.sort_values(['A', 'B'], ascending=[True, False])
df.sort_index()               # Sort by index
df.nlargest(5, 'col')         # Top 5 values
df.nsmallest(5, 'col')        # Bottom 5 values
```

## GroupBy Operations

```python
df.groupby('col').mean()      # Group and aggregate
df.groupby('col').sum()
df.groupby('col').count()
df.groupby('col').agg(['mean', 'sum', 'count'])
df.groupby(['col1', 'col2']).mean()  # Multiple columns

# Custom aggregation
df.groupby('col').agg({
    'A': 'sum',
    'B': 'mean',
    'C': ['min', 'max']
})

# Apply custom function
df.groupby('col').apply(lambda x: x.max() - x.min())
```

## Merging & Joining

```python
# Merge (SQL-like)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')
pd.merge(df1, df2, on='key', how='left')  # left, right, inner, outer

# Join (on index)
df1.join(df2)
df1.join(df2, how='outer')

# Concatenate
pd.concat([df1, df2])         # Vertically
pd.concat([df1, df2], axis=1) # Horizontally
```

## Pivot & Reshape

```python
# Pivot
df.pivot(index='A', columns='B', values='C')
df.pivot_table(values='C', index='A', columns='B', aggfunc='mean')

# Melt (unpivot)
pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])

# Stack/Unstack
df.stack()                     # Pivot columns to rows
df.unstack()                   # Pivot rows to columns

# Transpose
df.T
```

## String Operations

```python
df['col'].str.lower()         # Lowercase
df['col'].str.upper()         # Uppercase
df['col'].str.len()           # String length
df['col'].str.strip()         # Remove whitespace
df['col'].str.split(',')      # Split string
df['col'].str.contains('text') # Check contains
df['col'].str.replace('old', 'new')  # Replace
df['col'].str.startswith('A') # Starts with
df['col'].str.extract(r'(\d+)') # Extract pattern
```

## DateTime Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour

# Date arithmetic
df['date'] + pd.Timedelta(days=7)
df['date'] - pd.Timedelta(hours=3)

# Date range
pd.date_range('2024-01-01', '2024-12-31', freq='D')
pd.date_range('2024-01-01', periods=10, freq='M')

# Resample time series
df.resample('M').mean()        # Monthly mean
df.resample('W').sum()         # Weekly sum
```

## Missing Data

```python
df.isna()                      # Check for NaN
df.notna()                     # Check for not NaN
df.isnull()                    # Alias for isna()
df.notnull()                   # Alias for notna()

df.dropna()                    # Drop rows with NaN
df.dropna(axis=1)              # Drop columns with NaN
df.dropna(how='all')          # Drop only if all NaN
df.dropna(thresh=2)           # Keep rows with ≥2 non-NaN

df.fillna(0)                   # Fill NaN with value
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill
df.fillna(df.mean())          # Fill with mean
df.interpolate()              # Interpolate NaN
```

## Apply Functions

```python
# Apply to column
df['col'].apply(lambda x: x * 2)

# Apply to DataFrame
df.apply(lambda x: x.max() - x.min())
df.apply(np.sum)               # Apply numpy function

# Apply row-wise
df.apply(lambda row: row['A'] + row['B'], axis=1)

# Map values
df['col'].map({1: 'A', 2: 'B', 3: 'C'})

# Element-wise (applymap)
df.applymap(lambda x: x * 2)   # Deprecated, use .map()
df.map(lambda x: x * 2)        # Element-wise on DataFrame
```

## Statistics

```python
df.sum()                       # Sum of each column
df.mean()                      # Mean
df.median()                    # Median
df.mode()                      # Mode
df.std()                       # Standard deviation
df.var()                       # Variance
df.min()                       # Minimum
df.max()                       # Maximum
df.quantile(0.25)             # Quantile
df.corr()                      # Correlation matrix
df.cov()                       # Covariance matrix
df.value_counts()             # Count unique values
df.nunique()                   # Number of unique values
```

## Filtering

```python
df.query('A > 5 and B < 10')  # Query string
df.filter(items=['A', 'B'])   # Filter columns
df.filter(like='col')         # Columns containing 'col'
df.filter(regex='^col')       # Columns matching regex
df.select_dtypes(include=['int64'])  # Select by dtype
df.select_dtypes(exclude=['object']) # Exclude by dtype
```

## Categorical Data

```python
df['col'] = df['col'].astype('category')
df['col'].cat.categories      # View categories
df['col'].cat.codes           # Integer codes
df['col'] = df['col'].cat.reorder_categories(['A', 'B', 'C'])
df['col'] = df['col'].cat.rename_categories({'A': 'Alpha'})
```

## MultiIndex

```python
# Create MultiIndex
df.set_index(['A', 'B'])
pd.MultiIndex.from_tuples([('A', 1), ('A', 2)])

# Select from MultiIndex
df.loc[('A', 1)]
df.xs('A', level=0)

# Reset index
df.reset_index()
df.reset_index(drop=True)
```

## Performance Tips

```python
# Vectorized operations (fast)
df['C'] = df['A'] + df['B']

# Avoid loops (slow)
for i in df.index:
    df.loc[i, 'C'] = df.loc[i, 'A'] + df.loc[i, 'B']

# Use built-in methods
df.sum()  # Fast
sum(df['col'])  # Slower

# Copy vs view
df_copy = df.copy()           # True copy
df_view = df                  # Reference
```

## Useful Options

```python
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 2)
pd.reset_option('all')        # Reset to defaults
```
