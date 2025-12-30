# README.md

# 📊 Complete Pandas Tutorial: From Fundamentals to Advanced Data Manipulation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive, hands-on learning resource for mastering **Pandas**, Python's premier data manipulation and analysis library. This repository bridges the gap between basic syntax and real-world data engineering skills through structured tutorials, practical exercises, and production-ready techniques.

---

## What You'll Learn

This repository provides **progressive learning paths** covering:

### **Foundation Layer (Beginner)**
- **Data Structures Architecture**: Deep understanding of Series and DataFrame internals, index objects, and memory layout
- **DataFrame Construction**: Multiple methods to create DataFrames from dictionaries, lists, arrays, and external sources
- **Data Types & Casting**: Working with dtypes, categorical data, datetime objects, and efficient type conversions
- **Indexing Paradigms**: Mastering loc, iloc, at, iat, and boolean indexing patterns

### **Data Engineering Layer (Intermediate)**
- **ETL Operations**: Extract, Transform, Load workflows using Pandas I/O tools (CSV, Excel, JSON, Parquet, SQL)
- **Data Cleaning Pipeline**: Handling missing values with fillna, interpolation, forward/backward fill strategies
- **String Operations**: Vectorized string methods for text processing and pattern matching
- **Time Series Analysis**: Date ranges, resampling, rolling windows, and time-based indexing
- **Merge Strategies**: Inner, outer, left, right joins, concatenation, and handling merge conflicts

### **Advanced Analytics Layer (Advanced)**
- **GroupBy Mechanics**: Split-apply-combine paradigm, custom aggregations, transformation vs filtering
- **Window Functions**: Rolling, expanding, and exponentially weighted calculations
- **Performance Optimization**: Memory profiling, vectorization techniques, and avoiding common anti-patterns


---

## Repository Architecture

```
sidcancode/Pandas/
│
├── PlayPandas.ipynb              # Main interactive tutorial notebook
│                                 # Contains progressive exercises with solutions
│
├── data/                         # Curated datasets for practice
│   ├── bios.csv                  
│   ├── olympics-data.csv           # Time-series olympics records
│         
│
├── warmup-data/                  # Beginner-friendly practice files
│   ├── cofee.csv                 # Clean data for basic operations
│   
│
├── pandas-cheat-sheet.md.html   # Quick reference guide
│                                 # Searchable HTML cheat sheet
│
├── requirements.txt              # Pinned dependencies for reproducibility
└── .gitignore                    # Clean repository management
```

---

## Core Tutorial: PlayPandas.ipynb

The **PlayPandas.ipynb** notebook is structured as a **progressive learning journey**:

### Module Breakdown

1. **Pandas Fundamentals (Cells 1-15)**
   - Series vs DataFrame architecture
   - Creating data structures from scratch
   - Understanding the Index object
   - Basic selection and slicing

2. **Data Import/Export (Cells 16-25)**
   - Reading various file formats
   - Handling encoding issues
   - Working with delimiters and headers
   - Exporting to different formats

3. **Data Exploration (Cells 26-40)**
   - Statistical summaries with describe()
   - Data profiling techniques
   - Identifying data quality issues
   - Visualizing distributions

4. **Data Transformation (Cells 41-65)**
   - Column operations and calculations
   - Applying functions with apply() and map()
   - Reshaping with pivot and melt
   - Handling categorical variables

5. **Advanced Operations (Cells 66-90)**
   - Complex groupby operations
   - Multi-level aggregations
   - Window functions for rolling analytics
   - Merging complex datasets


---

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **Basic Python Knowledge**: Variables, functions, loops, list comprehensions
- **Optional**: Basic statistics and data concepts

### Installation Guide

#### Option 1: Local Setup (Recommended for Learning)

1. **Clone the Repository**
```bash
git clone https://github.com/siddharajD/sidcancode.git
cd sidcancode/Pandas
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

The requirements.txt includes:
```
pandas>=2.0.0
numpy>=1.24.0
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0      # Excel support
pyarrow>=12.0.0       # Parquet support
```

4. **Launch Jupyter**
```bash
jupyter notebook
```

Navigate to `PlayPandas.ipynb` and start learning!

#### Option 2: Cloud-Based Learning (No Installation)

**Google Colab** (Free, GPU-enabled):
1. Visit [colab.research.google.com](https://colab.research.google.com)
2. File → Upload Notebook → Select `PlayPandas.ipynb`
3. Run cells directly in browser

**Kaggle Notebooks**:
1. Visit [kaggle.com/notebooks](https://www.kaggle.com/notebooks)
2. Upload notebook and datasets
3. Access community kernels and discussions

---

## Technical Deep Dive: Key Concepts Covered

### 1. DataFrame Internals & Memory Management

Understanding how Pandas stores and accesses data:

```python
# Memory-efficient column types
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')

# View memory usage
df.memory_usage(deep=True)
df.info(memory_usage='deep')
```

**Learning Outcomes**:
- Reduce DataFrame memory footprint by 70-90%
- Choose optimal data types for columns
- Understand copy vs view in slicing operations

### 2. Vectorization vs Iteration

Why loops are slow and how to avoid them:

```python
# Slow: Iterative approach
for idx in df.index:
    df.loc[idx, 'new_col'] = df.loc[idx, 'col1'] * df.loc[idx, 'col2']

# Fast: Vectorized approach
df['new_col'] = df['col1'] * df['col2']

# Faster: NumPy operations when possible
df['new_col'] = df['col1'].values * df['col2'].values
```

**Learning Outcomes**:
- Achieve 10-100x speed improvements
- Write production-ready data processing code
- Understand when to use apply() vs vectorization

### 3. Advanced Groupby Patterns

Multi-dimensional aggregations:

```python
# Named aggregations (Pandas 0.25+)
result = df.groupby('category').agg(
    total_sales=('sales', 'sum'),
    avg_price=('price', 'mean'),
    customer_count=('customer_id', 'nunique')
)

# Custom aggregation functions
def q95(x):
    return x.quantile(0.95)

df.groupby('region')['revenue'].agg(['mean', 'std', q95])

# Transform for group-wise operations
df['sales_pct'] = df.groupby('store')['sales'].transform(lambda x: x / x.sum())
```

**Learning Outcomes**:
- Build complex analytical queries
- Implement custom business logic in aggregations
- Understand split-apply-combine paradigm

### 4. Time Series Mastery

Working with temporal data:

```python
# Proper datetime parsing
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index('date', inplace=True)

# Resampling time series
daily_data.resample('W').agg({
    'sales': 'sum',
    'customers': 'mean',
    'returns': 'sum'
})

# Rolling window calculations
df['7day_ma'] = df['sales'].rolling(window=7).mean()
df['30day_std'] = df['sales'].rolling(window=30).std()

# Time-based filtering
df['2024-01':'2024-06']  # First half of 2024
df.between_time('09:00', '17:00')  # Business hours only
```

**Learning Outcomes**:
- Process financial and IoT time series data
- Calculate moving averages and volatility metrics
- Handle timezone-aware operations

### 5. Data Cleaning Production Patterns

Building robust ETL pipelines:

```python
def clean_dataframe(df):
    """Production-ready data cleaning pipeline"""
    
    # 1. Handle missing values strategically
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 2. Standardize string columns
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].apply(lambda x: x.str.strip().str.lower())
    
    # 3. Remove duplicates with logging
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # 4. Validate data ranges
    assert df['age'].between(0, 120).all(), "Invalid age values found"
    
    return df
```

**Learning Outcomes**:
- Build reusable data processing functions
- Implement data quality checks
- Handle edge cases gracefully

---

## Pandas Cheat Sheet: Quick Reference

### Essential Operations

#### DataFrame Creation
```python
# From dictionary
df = pd.DataFrame({'col1': [1,2,3], 'col2': ['a','b','c']})

# From CSV with options
df = pd.read_csv('data.csv', 
                 parse_dates=['date_col'],
                 dtype={'id': 'int32'},
                 na_values=['NA', 'missing'])

# From SQL
df = pd.read_sql('SELECT * FROM table', connection)
```

#### Data Inspection
```python
df.head(10)                    # First 10 rows
df.tail(10)                    # Last 10 rows
df.sample(5)                   # Random 5 rows
df.info()                      # Column types and nulls
df.describe()                  # Statistical summary
df.shape                       # Dimensions (rows, cols)
df.columns                     # Column names
df.dtypes                      # Data types
df.isnull().sum()             # Count nulls per column
df.nunique()                   # Unique values per column
```

#### Selection & Filtering
```python
# Column selection
df['col1']                     # Single column → Series
df[['col1', 'col2']]          # Multiple columns → DataFrame

# Row selection
df.loc[0]                      # By label
df.iloc[0]                     # By position
df.loc[0:5, 'col1':'col3']    # Label-based slicing
df.iloc[0:5, 0:3]             # Position-based slicing

# Boolean filtering
df[df['age'] > 25]            # Simple condition
df[(df['age'] > 25) & (df['city'] == 'NYC')]  # Multiple conditions
df[df['name'].str.contains('John', case=False)]  # String matching
df.query("age > 25 and city == 'NYC'")  # SQL-style filtering
```

#### Data Transformation
```python
# Add/modify columns
df['new_col'] = df['col1'] * 2
df = df.assign(new_col2=lambda x: x['col1'] + x['col2'])

# Remove columns/rows
df.drop(columns=['col1'])
df.drop(index=[0, 1])
df.drop_duplicates()

# Rename
df.rename(columns={'old': 'new'}, inplace=True)

# Sort
df.sort_values('col1', ascending=False)
df.sort_values(['col1', 'col2'])
```

#### Aggregation & Grouping
```python
# Simple aggregations
df['col1'].sum()
df['col1'].mean()
df['col1'].median()
df['col1'].std()

# Groupby operations
df.groupby('category')['sales'].sum()
df.groupby('category').agg({
    'sales': ['sum', 'mean'],
    'quantity': 'count'
})

# Pivot tables
df.pivot_table(values='sales', 
               index='region', 
               columns='product', 
               aggfunc='sum')
```

#### Merging & Joining
```python
# Merge (SQL-style joins)
pd.merge(df1, df2, on='id', how='inner')  # inner, left, right, outer
pd.merge(df1, df2, left_on='id1', right_on='id2')

# Concatenate
pd.concat([df1, df2])                     # Vertical stack
pd.concat([df1, df2], axis=1)            # Horizontal stack
```

#### Missing Data
```python
df.fillna(0)                   # Fill with constant
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill
df.fillna(df.mean())           # Fill with mean
df.dropna()                    # Drop rows with any null
df.dropna(subset=['col1'])     # Drop rows where col1 is null
df.interpolate()               # Interpolate missing values
```

#### Time Series
```python
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resampling
df.resample('D').mean()        # Daily average
df.resample('W').sum()         # Weekly sum
df.resample('M').agg({'sales': 'sum', 'customers': 'mean'})

# Rolling windows
df['ma_7'] = df['value'].rolling(7).mean()
df['std_30'] = df['value'].rolling(30).std()

# Shifting
df['prev_day'] = df['value'].shift(1)
df['next_day'] = df['value'].shift(-1)
```

---


## Pro Tips for Learning

1. **Type Code Manually**: Don't copy-paste. Muscle memory accelerates learning.

2. **Experiment with Parameters**: Change function arguments to see different behaviors.

3. **Read Error Messages**: Pandas errors are descriptive—they teach you what went wrong.

4. **Use Documentation**: Press `Shift+Tab` in Jupyter for inline docs.

5. **Benchmark Your Code**: Use `%%timeit` to compare different approaches.

---

## After Completing This Tutorial

You'll be ready to:

**Build production ETL pipelines** for data warehousing  
**Perform exploratory data analysis** on real business datasets  
**Prepare data for machine learning** models  
**Create automated reporting systems**  
**Handle complex data transformations** with confidence  
**Debug Pandas code** efficiently  
**Optimize DataFrame operations** for large datasets

---

## 🌟 Recommended Next Steps: Advanced Topics

After mastering this tutorial, explore:

1. **Pandas Performance**: 
   - Dask for parallel computing
   - Modin for drop-in performance boost
   - Polars as a Rust-based alternative

2. **Data Visualization**:
   - Matplotlib integration
   - Seaborn for statistical plots
   - Plotly for interactive dashboards

3. **Machine Learning Integration**:
   - Scikit-learn preprocessing
   - Feature engineering techniques
   - Train-test splits with stratification

4. **Big Data Tools**:
   - PySpark DataFrames
   - SQL integration (SQLAlchemy)
   - Apache Arrow for data exchange

---

## Contributing

Found a bug or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new exercise'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Additional Resources

### Official Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)

### Recommended Reading
- "Python for Data Analysis" by Wes McKinney (Pandas creator)
- "Effective Pandas" by Matt Harrison


<div align="center">

**⭐ If you find this tutorial helpful, please star the repository! ⭐**

*Happy Data Wrangling! 🐼*

</div>
