### PIBO.py

### How to run:
  `$ python PIBO.py data_sample.csv`

### Input CSV format:
  The input CSV file must contain four numeric columns:
    column 1: target value y
    column 2: x1
    column 3: x2
    column 4: x3

### Optimization setting:
  - Optimization mode: minimization
  - Search range:
      x1 in [0.5, 1.5]
      x2 in [600, 900]
      x3 in [5, 25]
  - Fine grid size: 50 points per dimension
  - y-range: automatically determined from the minimum and maximum y values in the input data
  - Acquisition function: weighted expected improvement with center = 1.0,
    tau = 0.1, sigma = 0.1, epsilon = 0.3
  - Gaussian process kernel:
      ConstantKernel * Matern(nu = 2.5) + WhiteKernel

### Output:
  The script prints the recommended next point and related optimization information to the console.

### Requirements:
  - `python`: `3.13`
  - `numpy`: `2.4`
  - `pandas`: `2.3.3`
  - `scipy`: `1.16`
  - `scikit-learn`: `1.7`

### Example:
  `$ python PIBO.py data_sample.csv`
