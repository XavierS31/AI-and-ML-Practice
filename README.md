#  AI & Machine Learning Practice Repository

This repository contains a collection of **practice exercises** implementing core **Artificial Intelligence and Machine Learning** concepts **from scratch**. Each module focuses on a fundamental algorithm or technique, using Jupyter notebooks and simple datasets to reinforce theory through direct implementation.

The goal is to build strong intuition, mathematical understanding, and practical coding skills without relying on high-level libraries for the models themselves.

---

## 📁 Repository Structure

### **`AI_LinearRegression/`**

Practice projects for **univariate and multivariate linear regression**.
Includes:

* Gradient Descent
* Closed-Form (Normal Equation) Solution
* Cost Function Visualization
* Parameter update mechanics
* Dataset: `D3.csv`

### **`AI_ML_GaussianNaiveBayes/`**

Exercises exploring **Naïve Bayes classification**, both categorical and Gaussian variants.
Includes:

* Categorical NB on small animal dataset
* Gaussian NB on digit images
* Manual priors, likelihoods, and posteriors
* Custom implementation of Gaussian NB

### **`AI_ML_Logistic_Regression/`**

Practice implementation of **binary logistic regression with L2 regularization** using only NumPy.
Includes:

* Stratified splitting
* Standardization
* Sigmoid model & cross-entropy loss
* Vectorized gradient descent
* Full evaluation metrics

### **`PCA_CRC/`**

Exercises applying **Principal Component Analysis (PCA)** and **Collaborative Representation Classification (CRC)** for face recognition.
Includes:

* CRC with ℓ₂-regularization
* PCA via SVD for dimensionality reduction
* Performance comparison with and without PCA

### **`SVM/`**

Practice experiments with **Support Vector Machines**, comparing Linear and RBF kernels, with and without PCA.
Includes:

* Hyperparameter tuning for C and γ
* PCA for variance retention and speedup
* Confusion matrices and accuracy evaluation

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone <your_repo_url>
cd <repo_folder>
```

Install general dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Launch any notebook:

```bash
jupyter notebook
```

---

## 🎯 Purpose of This Repository

This project exists to reinforce foundational AI/ML concepts through:

* Clean, minimal implementations
* Experiments showing hyperparameter effects
* Mathematical intuition behind algorithms
* Practical application on real datasets like LFW

Ideal for students, self-learners, or anyone improving ML fundamentals.

---

## 📌 Included Concepts

* Linear Regression
* Logistic Regression
* Naïve Bayes (Categorical & Gaussian)
* PCA (Dimensionality Reduction)
* CRC (Face Recognition)
* Support Vector Machines
* Model evaluation & visualization

---

## 📄 Notes

* All implementations emphasize **clarity over optimization**.
* These are **practice exercises**, not production-ready models.
* More modules may be added over time.
