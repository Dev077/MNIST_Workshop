# Matplotlib Cheatsheet

## Import
```python
import matplotlib.pyplot as plt
import numpy as np
```

## Basic Plotting

### Line Plot
```python
plt.plot(x, y)                 # Basic line plot
plt.plot(x, y, 'r--')          # Red dashed line
plt.plot(x, y, label='Data')   # With label
plt.show()
```

### Scatter Plot
```python
plt.scatter(x, y)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
```

### Bar Plot
```python
plt.bar(x, height)             # Vertical bars
plt.barh(y, width)             # Horizontal bars
```

### Histogram
```python
plt.hist(data, bins=20, alpha=0.7)
plt.hist(data, bins=20, density=True)  # Normalized
```

### Box Plot
```python
plt.boxplot(data)
plt.boxplot([data1, data2, data3], labels=['A', 'B', 'C'])
```

### Pie Chart
```python
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
```

## Plot Customization

### Labels & Title
```python
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()                   # Show legend
plt.legend(loc='upper right')  # Legend position
plt.grid(True)                 # Add grid
```

### Axes & Limits
```python
plt.xlim(0, 10)               # X-axis limits
plt.ylim(0, 100)              # Y-axis limits
plt.axis([0, 10, 0, 100])     # Set both axes
plt.axis('equal')             # Equal aspect ratio
plt.axis('off')               # Hide axes
```

### Ticks
```python
plt.xticks([0, 2, 4, 6, 8])   # Set tick positions
plt.xticks([0, 2, 4], ['A', 'B', 'C'])  # Custom labels
plt.xticks(rotation=45)        # Rotate labels
plt.tick_params(labelsize=12)  # Tick label size
```

### Colors & Styles
```python
# Line styles: '-', '--', '-.', ':'
# Colors: 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'
# Markers: 'o', 's', '^', 'v', '*', '+', 'x'

plt.plot(x, y, 'ro-')          # Red circles with line
plt.plot(x, y, color='blue', linestyle='--', linewidth=2)
plt.plot(x, y, marker='o', markersize=5)
```

### Figure Size
```python
plt.figure(figsize=(10, 6))    # Width, height in inches
plt.figure(dpi=100)            # Resolution
```

## Subplots

### Multiple Subplots
```python
# Method 1: plt.subplot(rows, cols, index)
plt.subplot(2, 2, 1)           # 2x2 grid, 1st subplot
plt.plot(x, y1)
plt.subplot(2, 2, 2)           # 2nd subplot
plt.plot(x, y2)

# Method 2: plt.subplots()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y1)
axes[0, 1].plot(x, y2)
axes[1, 0].plot(x, y3)
axes[1, 1].plot(x, y4)
plt.tight_layout()             # Adjust spacing
```

### Sharing Axes
```python
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
```

## Object-Oriented Interface

```python
fig, ax = plt.subplots()

# Plotting
ax.plot(x, y)
ax.scatter(x, y)
ax.bar(x, height)
ax.hist(data, bins=20)

# Customization
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True)

# Twin axis (two y-axes)
ax2 = ax.twinx()
ax2.plot(x, y2, 'r')
```

## Advanced Plots

### Error Bars
```python
plt.errorbar(x, y, yerr=errors, fmt='o')
plt.errorbar(x, y, xerr=x_err, yerr=y_err, capsize=5)
```

### Fill Between
```python
plt.fill_between(x, y1, y2, alpha=0.3)
plt.fill_between(x, 0, y, where=(y > threshold), alpha=0.3)
```

### Contour Plot
```python
plt.contour(X, Y, Z)           # Contour lines
plt.contourf(X, Y, Z)          # Filled contours
plt.colorbar()                 # Add colorbar
```

### Heatmap (using imshow)
```python
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
```

### 3D Plots
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, z)
ax.scatter3D(x, y, z)
ax.plot_surface(X, Y, Z, cmap='viridis')
```

## Styling

### Built-in Styles
```python
plt.style.use('ggplot')        # ggplot style
plt.style.use('seaborn')       # seaborn style
plt.style.available            # List available styles
```

### Color Maps
```python
# Sequential: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
# Diverging: 'RdBu', 'coolwarm', 'bwr'
# Qualitative: 'tab10', 'tab20', 'Paired'

plt.scatter(x, y, c=values, cmap='viridis')
plt.colorbar()
```

### Text & Annotations
```python
plt.text(x, y, 'Text', fontsize=12)
plt.annotate('Point', xy=(x, y), xytext=(x+1, y+1),
            arrowprops=dict(arrowstyle='->'))
```

## Saving Figures

```python
plt.savefig('plot.png')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf')        # Vector format
plt.savefig('plot.svg')        # Scalable vector
```

## Multiple Figures

```python
fig1 = plt.figure(1)
plt.plot(x, y1)

fig2 = plt.figure(2)
plt.plot(x, y2)

plt.figure(1)                  # Switch to figure 1
plt.close()                    # Close current figure
plt.close('all')               # Close all figures
```

## Customizing Defaults

```python
# Temporarily change settings
with plt.rc_context({'lines.linewidth': 2, 'lines.linestyle': '--'}):
    plt.plot(x, y)

# Global settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
```

## Common Plot Types Cheatsheet

```python
# Line: plt.plot(x, y)
# Scatter: plt.scatter(x, y)
# Bar: plt.bar(x, height)
# Histogram: plt.hist(data)
# Box: plt.boxplot(data)
# Pie: plt.pie(sizes)
# Errorbar: plt.errorbar(x, y, yerr=err)
# Stem: plt.stem(x, y)
# Step: plt.step(x, y)
# Violin: plt.violinplot(data)
```

## Tips & Tricks

```python
# Clear current figure
plt.clf()

# Clear current axes
plt.cla()

# Interactive mode
plt.ion()                      # Turn on
plt.ioff()                     # Turn off

# Pause to update plot
plt.pause(0.1)

# Get current figure/axes
fig = plt.gcf()
ax = plt.gca()

# Log scale
plt.xscale('log')
plt.yscale('log')
plt.loglog(x, y)               # Both axes log scale

# Invert axis
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

# Axis spine customization
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

## Date/Time Plots

```python
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ax.plot(dates, values)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
```
