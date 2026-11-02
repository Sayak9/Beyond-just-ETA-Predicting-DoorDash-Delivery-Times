# Beyond just ETA: Predicting DoorDash Delivery Times

When someone places a food order, one of the first things they want to know is how long it will take to arrive. Getting that estimate right is important. A prediction that is too optimistic can lead to a frustrating experience, while one that is too conservative may not be particularly useful either.

This project explores how machine learning can be used to predict the total time taken for a food delivery, from the moment an order is placed to when it reaches the customer.

Using historical DoorDash order data, I work through data exploration, feature engineering, feature selection and multicollinearity checks before comparing several regression algorithms.

The most interesting part of the project is the change in modelling strategy. Rather than predicting the entire delivery duration directly, I experiment with breaking it into smaller components. A LightGBM model first estimates preparation time, and a second regression model combines that prediction with the estimated driving and order placement durations to predict the total delivery time.

## Project Objective

The goal is to predict the total delivery duration, measured in seconds.

The duration is calculated as the difference between two timestamps:

1. **Start:** `created_at`, when the customer submits the order.
2. **End:** `actual_delivery_time`, when the order reaches the customer.

This makes the problem a supervised regression task, where historical delivery records are used to train models to estimate future delivery durations.

## Dataset

The project uses `historical_data.csv`, containing approximately 197,428 historical delivery records and 16 original columns.

The dataset includes information about:

1. Order details, including the number of items and order subtotal.
2. Store information, including market, store identifier and primary food category.
3. Order placement protocol.
4. Delivery operations, including the number of dashers on shift, busy dashers and outstanding orders.
5. Estimated order placement duration.
6. Estimated driving duration from the store to the customer.
7. Actual order creation and delivery timestamps.

The target variable, `actual_total_delivery_duration`, is created by calculating the difference between the actual delivery and order creation timestamps.

After removing missing and infinite values, the initial modelling table contains 177,070 records.

## Tools and Technologies

1. **Python:** Overall data analysis and model development.
2. **Pandas:** Data manipulation, cleaning, feature engineering and aggregation.
3. **NumPy:** Numerical operations and data preparation.
4. **Matplotlib and Seaborn:** Correlation analysis and visualisation.
5. **Scikit Learn:** Data preprocessing, feature selection, regression models and evaluation.
6. **Statsmodels:** Multicollinearity analysis using Variance Inflation Factor.
7. **XGBoost:** Gradient boosting regression.
8. **LightGBM:** Gradient boosting and preparation time prediction.
9. **TensorFlow and Keras:** Artificial neural network experimentation.

## Methodology

### 1. Data Exploration and Target Creation

The analysis begins with a closer look at the dataset, its column types, missing values and descriptive statistics.

The original timestamps are converted into datetime objects, and their difference is used to calculate the actual delivery duration in seconds.

This newly created column becomes the target variable for all subsequent regression experiments.

### 2. Feature Engineering

Before training any models, I explored whether the existing columns could be combined to provide more useful information.

Three particularly interesting features were created:

**Estimated non preparation duration**

Combines the estimated order placement duration and the estimated store to customer driving duration.

**Busy dasher ratio**

Calculated by dividing the number of busy dashers by the number of dashers currently on shift. This provides a simple indication of delivery capacity utilisation.

**Store category imputation**

Missing store categories are filled using the most frequently recorded category associated with each store. Where multiple categories are tied, the first returned category is used.

The categorical variables are subsequently converted into numerical representations using one hot encoding.

### 3. Data Cleaning and Feature Preparation

Identifiers and original text columns that are not directly useful for the regression models are removed after encoding the relevant categorical information.

Infinite values are replaced with missing values, and incomplete records are removed to produce a clean numerical dataset.

At this stage, the initial modelling table contains 100 columns, including the target variable.

### 4. Correlation Analysis and Feature Engineering

With a relatively large number of candidate features, I wanted to identify variables that were repeating similar information.

A correlation matrix and helper functions are used to identify the strongest relationships between feature pairs.

A few notable relationships emerge:

1. The number of busy dashers and the number of dashers on shift are strongly correlated.
2. Estimated driving and order placement durations are closely related to the engineered non preparation duration.
3. The number of items, distinct items and order subtotal share overlapping information.
4. Minimum and maximum item prices are also correlated.

Instead of simply removing every correlated variable, I experimented with creating more informative features.

These include:

1. Percentage of distinct items relative to total items.
2. Average price per item.
3. Price range between the most expensive and least expensive items.

This reduced the modelling table to 82 columns while retaining features that could contribute useful information about preparation time.

### 5. Multicollinearity Analysis Using VIF

Pairwise correlation does not always reveal the full extent of multicollinearity, so I used Variance Inflation Factor to investigate the issue further.

The iterative approach removes the feature with the highest VIF and recalculates the remaining scores until all predictors fall below a threshold of 20.

The `percent_distinct_item_of_total` feature was removed because its VIF exceeded the selected threshold.

This left a more manageable feature set for the modelling experiments.

### 6. Feature Selection and PCA

A Random Forest Regressor was trained to understand which features contributed most to predicting delivery duration.

The feature importance scores were used to create several alternative feature sets containing:

1. All available predictors.
2. The 40 most important features.
3. The 20 most important features.
4. The 10 most important features.

I also explored Principal Component Analysis to see whether the feature dimensions could be reduced further.

However, approximately 60 components were needed to explain 80% of the variance. Since this offered relatively little reduction compared with the existing feature selection approach, PCA was not used in the final model.

### 7. Scaling and Classical Regression Models

The features have considerably different numerical ranges, so I experimented with three approaches:

1. StandardScaler.
2. MinMaxScaler.
3. No scaling.

I compared several regression algorithms across the different feature sets:

1. Ridge Regression.
2. Decision Tree Regressor.
3. Random Forest Regressor.
4. XGBoost Regressor.
5. LightGBM Regressor.
6. Multi Layer Perceptron Regressor.

Root Mean Squared Error was selected as the primary evaluation metric, allowing model errors to be compared in the original unit of measurement, seconds.

**What I found**

LightGBM consistently performed among the best models in the direct prediction experiments.

Using the full feature set with StandardScaler, the reference solution recorded an RMSE of approximately 2,033 seconds.

Reducing the feature count to 40 produced a very similar result, while using only 10 features resulted in a slight deterioration.

Although the models were performing reasonably consistently, their overall prediction errors were still substantial.

This prompted me to reconsider the way the problem was being approached.

## The Two Stage Modelling Approach

### 8. Predicting Preparation Time Separately

A delivery can be viewed as three main activities:

1. Order placement.
2. Food preparation.
3. Driving from the store to the customer.

The dataset already provides estimates for order placement and driving durations. However, it does not contain a separate preparation time estimate.

I therefore created a new target variable:

`prep_time = actual_total_delivery_duration - estimated_driving_duration - estimated_order_placement_duration`

The objective was to predict this preparation component independently rather than trying to estimate the entire delivery duration in one step.

The same family of regression models was evaluated, using the 40 most important features while excluding the two estimated durations that would be added back later.

LightGBM produced the strongest result in the supplied experiment, with an RMSE of approximately 2,036 seconds for preparation time prediction.

### 9. Reconstructing Total Delivery Duration

Once the preparation time predictions were available, I combined them with the estimated driving and order placement durations.

The first attempt was simply to add all three components together.

However, this produced an RMSE of approximately 2,036 seconds, which was very close to the direct prediction approach.

Rather than assuming that the three components should contribute equally, I tried a second regression model.

This model receives three inputs:

1. Predicted preparation duration.
2. Estimated driving duration.
3. Estimated order placement duration.

Its target is the actual total delivery duration.

This allows the model to learn how the three components relate to the final delivery time instead of relying on a straightforward addition.

### 10. Comparing the Second Stage Models

Several regression models were tested on the three duration components.

The results from the supplied guided solution were:

| Regression model | RMSE in seconds |
|---|---:|
| Linear Regression | 986.69 |
| Ridge Regression | 986.69 |
| MLP Regressor | 987.30 |
| LightGBM | 1,079.29 |
| Random Forest | 1,164.00 |
| Decision Tree | 1,235.58 |
| XGBoost | 1,370.41 |

Linear Regression and Ridge Regression performed particularly well, with almost identical results.

This was an interesting finding because the simpler linear models outperformed the more complex tree based and boosting models at this stage.

### 11. Artificial Neural Network Experiment

As a final experiment, I also tested a small Artificial Neural Network using TensorFlow and Keras.

The architecture consists of:

1. An input layer receiving the three duration features.
2. A hidden layer with 16 neurons and ReLU activation.
3. A single output neuron with linear activation.

The network was trained using Stochastic Gradient Descent, Mean Squared Error loss and 100 training epochs.

The supplied solution recorded an RMSE of approximately 998 seconds.

Although the neural network performed reasonably well, it did not improve upon Linear Regression.

For this particular problem, the additional complexity did not appear to offer a meaningful advantage.

## Results and Final Takeaway

The most interesting finding from this project was not simply which algorithm performed best, but how changing the structure of the prediction problem affected the results.

The direct modelling approach produced an RMSE of approximately 2,033 seconds using LightGBM.

The two stage approach, which first estimates preparation time and then combines the three duration components using a second regression model, achieved a substantially lower reported RMSE.

**Linear Regression was the strongest performer in the second stage, with an RMSE of 986.69 seconds.**

This highlights an important modelling idea: sometimes decomposing a complex prediction problem into smaller, more meaningful components can produce better results than trying to predict everything at once.

The reference scores above are from the supplied guided solution. Results from a fresh execution may differ because the revised notebook fits preprocessing scalers on training data only and uses its own model splits.

## Limitations and Further Exploration

There are a few areas that could be explored further.

1. The dataset is historical, so its patterns may not necessarily reflect current delivery operations.
2. Store category imputation can be ambiguous when a store is associated with multiple categories.
3. The original feature screening and VIF calculations are exploratory and should be moved inside a training pipeline for rigorous validation.
4. The model could benefit from additional hyperparameter tuning.
5. A time based validation split would provide a more realistic assessment of performance on future orders.
6. Further investigation into preparation time, restaurant workload and dasher availability could improve the feature engineering process.

The current notebook is intended as an exploratory machine learning project rather than a production ready delivery prediction system.

## How to Run the Project

The notebook was developed using Python and Jupyter Notebook.

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost lightgbm tensorflow
```

Place `historical_data.csv` in the same directory as the notebook.

Open `doordash_delivery_time_prediction_humanized.ipynb` and execute the cells sequentially.

Some model comparisons, particularly Random Forest and boosting algorithms, may take longer to complete.

## Dataset Attribution

Project context: DoorDash delivery time prediction.

Dataset: `historical_data.csv`, provided with the DoorDash guided exercise.

The analysis is based on historical order information and the accompanying delivery duration prediction problem.
