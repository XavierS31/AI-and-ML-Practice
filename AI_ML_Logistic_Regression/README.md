# Logistic Regression from Scratch 

This project implements **binary logistic regression with L2 regularization** from scratch using only `NumPy`.  

---

## 📌 Description

We train a logistic regression model on the **Breast Cancer Wisconsin dataset** from `scikit-learn`.  
The model is implemented entirely with NumPy operations — no scikit-learn estimators are used.  

The pipeline includes:

1. **Dataset loading**  
   - Breast Cancer dataset (569 samples, 30 features)  
   - Binary classification: malignant (1) vs. benign (0)  

2. **Stratified splitting**  
   - 60% training, 20% validation, 20% testing  
   - Preserves class balance across subsets  

3. **Feature standardization**  
   - Zero mean and unit variance per feature  
   - Calculated from training set only  

4. **Logistic regression model**  
   - Hypothesis:  
$$
\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x}), \quad 
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
   - Loss: Binary cross-entropy + L2 penalty  

5. **Gradient descent optimization**  
   - Vectorized gradient update  
   - Regularization excludes bias term  

6. **Evaluation metrics**  
   - Accuracy, Precision, Recall, F1 Score  
   - Confusion matrix  
   - Predicted probabilities for first 5 test examples  

---

## 📊 Final Test Results

- **Accuracy:** 0.974  
- **Precision:** 0.960  
- **Recall:** 1.000  
- **F1 Score:** 0.980  

**Confusion Matrix (TP, TN, FP, FN):** (72, 40, 3, 0)

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
