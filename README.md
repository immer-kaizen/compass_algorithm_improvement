# compass_algorithm_improvement
# 1. Problem Statement

We are investigating COMPAS, a popular commercial algorithm for predicting reoffending likelihoods (recidivism) for criminal defendants. There has been evidence that the algorithm is biased towards white defendants and against black inmates. Next, use a fairness approach to improve the model.
In order to accomplish this, I need first to explore the data and prepare it, then assess the fairness, then use one of the approaches to optimize it.

Here are the steps I will take in this project: 
    
     1. Data Collection
     2. Data Exploration: This will be done to identify the most important features and combine them in new ways.
     3. Data Preprocessing: Lay out a pipeline of tasks for transforming data for use in my machine learning model.
     4. Model Assessment: Determine the type of descrimination.
     6. How to improve the fairness 
     7. Conclusion & recommendations     
     
Example of what you will find in this project 
![image](https://user-images.githubusercontent.com/98276432/224685403-fcf502bf-6270-44f3-96f7-03cfc02703a1.png)

# Conclusion 
1. Compas evaluates by race and gender; there are some differences in results, but we must not forget that one of the races has the highest prevalence (nearly 50%) and they are also the most likely to recidivate (nearly 40%).
2. Adding the two_year_r feature based on compas report was for the purpose to see if adding this feature will help to narrow the prediction coming from COMPASS 
3. We saw that is_violent_racid got better compard to is_recid but no improvement on Positive cases prediction it became worse. 
My suggestion, is to remove the race from the full data and train the model in a way that it will be always independant from the race. In that way the discrimination will be avoided, maybe not fully but at least it would be better.   
