<h1><b>DDoS Detection System Using Machine Learning</h1></b>
<br>
<br>
<h6>Description:</h6>
<br>
<p>This project implements a machine learning-based Distributed Denial-of-Service (DDoS) detection system. It is designed to detect abnormal network traffic patterns using a combination of Random Forest classification and domain-specific rule-based analysis. The system can generate synthetic network traffic data, perform feature engineering, and train a model to classify whether the traffic represents a DDoS attack or normal behavior.</p>
<br>
<h6>Key features:</h6>
<br>
<ul>
  <li>Data Generation: Simulates normal and DDoS attack traffic patterns, including packet size, rate, entropy, and flow duration.</li>
<li>Feature Engineering: Adds advanced traffic metrics like inter-arrival time, rolling mean, and bytes per second to enhance model accuracy.</li>
  <li>Model Training: Utilizes Random Forest Classifier, with features selected via SelectKBest, to detect DDoS attacks. The model is trained on balanced data using SMOTE for oversampling minority attack data.</li>
  <li>Prediction and Scoring: Incorporates both probabilistic model predictions and rule-based scoring for enhanced detection reliability</li>
  <li>Evaluation Metrics: Provides detailed confusion matrix, classification report, and ROC AUC score for model evaluation.</li>
  <li>Explainability: Uses SHAP and permutation importance for explaining feature contributions in model predictions.</li>
</ul>
<br>
<b><h6>Technologies:</h6></b>
<br>
<ul>
  <li>Python</li>
  <li>NumPy, pandas for data manipulation
  </li>
  <li>scikit-learn for model building and evaluation</li>
  <li>SMOTE for data balancing</li>
  <li>SHAP for explainability</li>
  <li>Matplotlib, Seaborn for visualization</li>
  <li>Joblib for model serialization</li>
  <li>MLflow for experiment tracking</li>
  
</ul>
<br>
<h6><b>How to use:</b></h6>
<br>
<ol>
  <li>Generate or input your network traffic data.</li>
  <li>Train the model using the train_model() function.</li>
  <li>Use predict() to classify new traffic instances and get an alert level (Low, Medium, High).</li>
</ol>
