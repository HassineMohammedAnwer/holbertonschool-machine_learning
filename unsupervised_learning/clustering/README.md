Clustering
In the context of Gaussian Mixture Models (GMMs), **responsibilities** are also referred to as **posterior probabilities**. These represent the probability that a particular data point \( X_i \) belongs to a specific cluster \( k \), given the current model parameters (cluster priors, means, and covariances). Responsibilities are denoted as \( g_{k, i} \), where:

\[
g_{k, i} = P(Z_i = k \mid X_i)
\]

Here:
- \( Z_i \) is the latent variable that indicates the cluster assignment of the data point \( X_i \).
- \( g_{k, i} \) is the posterior probability for cluster \( k \) and data point \( X_i \).

---

### **Calculation of Responsibilities**

Responsibilities are calculated during the **Expectation (E-step)** of the EM algorithm using **Bayes' Theorem**:

\[
g_{k, i} = \frac{\pi_k \cdot \mathcal{N}(X_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(X_i \mid \mu_j, \Sigma_j)}
\]

Where:
- \( \pi_k \): Prior probability (weight) of cluster \( k \).
- \( \mathcal{N}(X_i \mid \mu_k, \Sigma_k) \): Multivariate Gaussian density for cluster \( k \), evaluated at data point \( X_i \).
- \( \sum_{j=1}^K \pi_j \cdot \mathcal{N}(X_i \mid \mu_j, \Sigma_j) \): Normalization term that ensures \( \sum_{k=1}^K g_{k, i} = 1 \).

---

### **Interpretation**

- **Responsibility \( g_{k, i} \):** A fractional assignment indicating how much cluster \( k \) is responsible for data point \( X_i \). 
  - Values range from 0 to 1.
  - If \( g_{1, i} = 0.9 \), it means cluster 1 is 90% responsible for \( X_i \).

- **Soft Assignments:** In GMM, clusters are **soft**: a data point can partially belong to multiple clusters. This is unlike K-means, where a point belongs entirely to one cluster.

---

### **Use in EM Algorithm**

- **E-Step:** Responsibilities \( g_{k, i} \) are calculated for every data point \( i \) and cluster \( k \) based on the current parameters of the GMM.
- **M-Step:** These responsibilities are used to update:
  - Cluster weights \( \pi_k \): Proportion of data points assigned to each cluster.
  - Cluster means \( \mu_k \): Weighted average of the data points.
  - Cluster covariances \( \Sigma_k \): Weighted covariance of the data points.

---

### **Example**

Suppose we have:
- 2 clusters (\( k = 2 \)).
- Data point \( X_i \).
- Priors \( \pi_1 = 0.6 \), \( \pi_2 = 0.4 \).
- Gaussian densities for \( X_i \):
  - \( \mathcal{N}(X_i \mid \mu_1, \Sigma_1) = 0.5 \).
  - \( \mathcal{N}(X_i \mid \mu_2, \Sigma_2) = 0.3 \).

The responsibilities for \( X_i \) are:
\[
g_{1, i} = \frac{0.6 \cdot 0.5}{(0.6 \cdot 0.5) + (0.4 \cdot 0.3)} = 0.714
\]
\[
g_{2, i} = \frac{0.4 \cdot 0.3}{(0.6 \cdot 0.5) + (0.4 \cdot 0.3)} = 0.286
\]

This means:
- Cluster 1 is 71.4% responsible for \( X_i \).
- Cluster 2 is 28.6% responsible for \( X_i \).

---