# Customer Predictive Analysis
Customer analytics is a process by which data from customer behavior is used to help make key business decisions via market segmentation and predictive analytics. This information is used by businesses for direct marketing, site selection, and customer relationship management. Marketing provides services in order to satisfy customers. With that in mind, the productive system is considered from its beginning at the production level, to the end of the cycle at the consumer. Customer analytics plays an important role in the prediction of customer behavior.

This repository presents a comprehensive guide on conducting customer predictive analysis using machine learning techniques with Dataiku. Customer predictive analysis is a crucial task for businesses aiming to anticipate customer behavior, enhance customer experience, and optimize marketing strategies. Leveraging the powerful capabilities of Dataiku, a leading AI and machine learning platform, this analysis provides insights into customer segmentation, churn prediction, lifetime value estimation, and recommendation systems.

![unsplash.jpg](/images/unsplash.jpg)

## Overview
The analysis begins with an overview of the dataset used, which typically comprises various customer-related attributes such as demographics, transaction history, browsing behavior, and engagement metrics. With Dataiku's intuitive interface, data preparation tasks such as cleaning, transformation, and feature engineering are streamlined, ensuring the dataset is well-prepared for modeling.

## Modeling
The modeling phase involves selecting appropriate machine learning algorithms based on the specific objectives of the analysis. Techniques such as classification, regression, clustering, and collaborative filtering are applied to build predictive models that can accurately forecast customer behavior. Dataiku's extensive library of machine learning algorithms and automated model training capabilities simplify this process, allowing for rapid experimentation and model iteration.

## Evaluation and Interpretation
Once the models are trained, they are evaluated using various performance metrics such as accuracy, precision, recall, and F1-score. Interpretability techniques such as feature importance analysis, SHAP values, and model visualization are employed to gain insights into the factors driving customer behavior predictions. This facilitates informed decision-making and strategy refinement for maximizing business outcomes.

## Retail
Although until recently over 90% of retailers had limited visibility on their customers, with increasing investments in loyalty programs, customer tracking solutions and market research, this industry started increasing use of customer analytics in decisions ranging from product, promotion, price and distribution management.[citation needed] The most obvious use of customer analytics in retail today is the development of personalized communications and offers and/or different marketing programs by segment.[citation needed] Additional reasons set forth by Bain & Co. include: prioritizing product development efforts, designing distribution strategies and determining product pricing.[3] Demographic, lifestyle, preference, loyalty data, behavior, shopper value and predictive behavior data points are key to the success of customer analytics.

## Retail Management
Companies can use data about customers to restructure retail management. This restructuring using data often occurs in dynamic scheduling and worker evaluations. Through dynamic scheduling, companies optimize staffing through predictive scheduling software based on predictive customer traffic.  Worker schedules can be adjusted in response to updated forecasts at short notice. Customer analytics allows retail companies to evaluate workers by comparing daily sales to daily traffic in a store.  The use of customer analytics data affecting the management of retail workers in a phenomenon known as refractive surveillance. The model of refractive surveillance describes how the collection of information on one group can affect and allow for the control of an entirely different group.


# Installation
On this repository, you may find my personal projects related to Machine Learning, EDA, Python Jupyter Notebook and couple of Visualization based on the Dataiku Platform exported standard files. Most of the datasets I've been working with, downloaded from Conundrum site. Installation pretty straight forward. Simply download the whole set as a single project and as a ZIP file, everything have been flattened out with plain text files, and no SQL dump was involved, so there wouldn't be any missing system dependencies issue. Simply imported the downloaded Zip file to your working project.

# Jupyter Notebooks

- [Correlations analysis on Customer_Analysis_prepared (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/Correlations%20analysis%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [Linear Programming Optimization on Customer_Analysis_prepared (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/Linear%20Programming%20Optimization%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [PCA on Customer_Analysis_prepared (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/PCA%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [PCA on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/PCA%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [Time series forecasting on Customer_Analysis_prepared (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/Time%20series%20forecasting%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [Time-Series analytics on Customer_Analysis_prepared (admin).ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/Time-Series%20analytics%20on%20Customer_Analysis_prepared%20(admin).ipynb)
- [notebook editor for compute_Python_Customer_Analysis.ipynb](https://github.com/leonism/Customer-Predictive-Analysis/blob/master/ipython_notebooks/notebook%20editor%20for%20compute_Python_Customer_Analysis.ipynb)


# Correlation Matrix

The very first correlation analysis consists of plotting the "Correlation matrix" for numerical variables. For each couple of numerical variables, this computes the "strength" of the correlation (called the Pearson coefficient):

- 1.0 means a perfect correlation
- 0.0 means no correlation
- -1.0 means a perfect "inverse" correlation

Since it does not really make sense to print this correlation plot for hundred of variables, we are restricting it to the first 50 numerical variables of the dataset.
![download-1.png](/images/download-1.png)

# Regression Analysis with excluding Non-significant variables
![download.png](/images/download.png)

# Regression Analysis with both Continuous and Categorical Variable
![download.png](/images/sales-and-response.png)

Been enjoying exploring this dataset for sure, and certainly it was fun doing it, stays safe everyone! 😊

# Conclusion
In conclusion, this repository serves as a practical guide for conducting customer predictive analysis with Dataiku, demonstrating the end-to-end process from data preparation to model deployment. By leveraging the combined power of machine learning and Dataiku's platform, businesses can gain actionable insights into customer behavior and drive sustainable growth and profitability.

# Disclaimer
And please remember, as this is only a weekend pet project, which I'm doing them for my personal interest only.
