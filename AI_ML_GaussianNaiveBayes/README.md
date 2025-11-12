# Naïve Bayes Practice – Categorical and Gaussian Models

This repo contains two Jupyter Notebooks exploring **Naïve Bayes classifiers** from scratch.  
The goal is to practice implementing both categorical and continuous versions without relying on pre-built sklearn models.

---

## Overview

- **Notebook 1 – Categorical Naïve Bayes**
  - Uses a small toy dataset of animals with four categorical features:
    1. Give Birth (yes/no)  
    2. Can Fly (yes/no)  
    3. Live in Water (no/sometimes/yes)  
    4. Have Legs (yes/no)  
  - Classes: Mammal vs Non-Mammal.  
  - Implements:
    - Class priors
    - Likelihoods with Laplace smoothing
    - Manual calculation of posteriors and MAP classification  
  - Includes a worked example showing each step:
    - Prior
    - Likelihood with smoothing
    - Evidence
    - Posterior

---

- **Notebook 2 – Gaussian Naïve Bayes**
  - Applies Naïve Bayes to digit classification.  
  - Each feature (pixel) is modeled as a Gaussian:
    $$
    P(x_i \mid C=c) = \frac{1}{\sqrt{2\pi\sigma^2_{ic}}} 
    \exp\!\left(-\frac{(x_i - \mu_{ic})^2}{2\sigma^2_{ic}}\right)
    $$
  - Implementation details:
    - From-scratch `GaussianNBStudent` class
    - Estimates per-class priors, means, and variances
    - Uses log-likelihoods to avoid underflow
    - Evaluates accuracy, confusion matrix, precision, recall, and F1-score

---

## Results (Summary)

- **Categorical Naïve Bayes**
  - Correctly classifies animals based on attribute patterns.
  - Demonstrates how Laplace smoothing avoids zero probabilities.

- **Gaussian Naïve Bayes**
  - Achieves ~56–63% accuracy on handwritten digits.
  - Performs well on some digits (0,1,6,9) but struggles on others (2,3,5,7,8).
  - Highlights the limitations of the independence assumption in complex datasets.

---

## How to Run

1. Clone this repo or download the notebooks.  
2. Install Python 3.9+ and required libraries:
   ```bash
   pip install numpy matplotlib
