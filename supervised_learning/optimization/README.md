# 0x03-optimization
## normalization_constants(X)
This function takes a numpy array X of shape (m, nx) and returns the mean and standard deviation of each feature, respectively. In other words, it calculates the normalization (standardization) constants of a matrix.
## normalize(X, m, s)
This function takes a numpy array X of shape (d, nx), as well as two numpy arrays m and s of shape (nx,) that contain the mean and standard deviation of each feature of X, respectively. The function returns a normalized version of X.
