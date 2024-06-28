# Weather or Not: Bay Area VTA Ridership Forecasting

## Abstract
Combining weather and transit data can significantly enhance public transportation systems' efficiency and sustainability. This project presents a novel approach by integrating NOAA weather data with VTA transit data to model and predict ridership at VTA stops. The problem was formulated as a regression task, leveraging features from VTA routes, geographical attributes, and daily weather conditions. Various regression models were trained and evaluated, with the Random Forest model achieving the best performance with an R² score of 0.8050. This model enables VTA authorities to predict ridership effectively and optimize transit planning.

## Keywords
- Machine Learning
- Weather Data
- Ridership Forecasting
- Cross-Validation
- Ensemble Models
- Hyperparameter Tuning

## Introduction
Integrating machine learning with transit data offers an opportunity to transform public transportation infrastructure. This project investigates the impact of weather on VTA network ridership using advanced machine learning techniques, aiming to enhance transportation operations and sustainability. A comprehensive dataset spanning four years was utilized to develop a robust prediction model. The structured machine learning pipeline includes feature engineering, model construction, exploratory data analysis (EDA), data integration and processing, and thorough model evaluation.

## Problem Statement
In the Bay Area, public transportation is crucial for sustainable urban mobility by reducing carbon emissions. However, reliability is often compromised by unpredictable weather, leading to delays and increased emissions from idle vehicles. This project aims to create a predictive model to forecast transit ridership based on weather impacts, thereby improving service reliability and sustainability.

## Literature Survey
- **Weather and Public Transportation**: Studies highlight the significant impact of weather on public transportation ridership and the benefits of using machine learning for predicting passenger load changes.
- **Particulate Matter and Traffic Data**: Research demonstrates the relationship between air quality, traffic, and meteorological conditions, providing insights for environmental impact assessments.
- **Machine Learning in Transportation**: Machine learning methodologies have been successfully applied to predict real-time delays in transportation, offering valuable insights for similar challenges in bus and light rail systems.
- **Computational Fluid Dynamics and Health Safety**: Studies explore the risks of airborne infections in public transportation, emphasizing the importance of health safety measures.

## Data Collection
### VTA Ridership Data
- **Source URL**: [VTA Open Data - Ridership Data](https://data.vta.org/documents/a3e899b0822e433fb52dda8a2d4f140c/about)
- **Data Range**: 2014 - 2017
- **Description**: Contains ridership numbers across various VTA services.

### NOAA Weather and Station Data
- **Source URL**: [NOAA Weather and Station Data](https://noaa-ghcn-pds.s3.amazonaws.com/index.html#csv/)
- **Data Range**: 2014 - 2017
- **Description**: Meteorological observations including temperature, precipitation, snowfall, and wind speeds, with metadata for each weather station.

## Exploratory Data Analysis
Visualizations and statistical analyses were performed to understand the data's unique features and trends. Key findings include:
- **Weekly and Daily Distribution**: Analyzing ridership distribution over weekdays and years to identify patterns.
- **Correlation Heat Map**: Identifying relationships between weather variables and ridership, such as the positive correlation between temperature variables and ridership numbers.

## Data Preprocessing
- **Ridership Data**: Cleaned and aggregated ridership data to ensure accurate daily predictions.
- **Station Data**: Filtered NOAA weather stations to include only those within the Bay Area.
- **Weather Data**: Selected high-quality weather data from relevant stations, reducing the need for imputing missing values.
- **Combining Datasets**: Integrated weather features with ridership data based on geographic proximity and date.

## Modeling
### Linear Regression (Elastic Net with PCA)
Linear Regression models were optimized using PCA and Elastic Net regularization to capture relevant features.

### Decision Tree Regression
Non-linear interactions were modeled using Decision Trees, with hyperparameter tuning to enhance performance.

### Random Forest Regression
Random Forest models were employed to reduce overfitting and improve prediction accuracy.

### Gradient Boosted Trees
Gradient Boosted Trees were used for their ability to iteratively refine predictions, focusing on residual errors.

### XGBoost
XGBoost was selected for its efficiency and scalability in handling large datasets and complex interactions.

## Evaluation
Models were evaluated using metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Variance Score, and R² score. The Random Forest model achieved the highest performance with an R² score of 0.8050.

## Impact
This project demonstrates the potential of integrating weather and transit data to improve public transportation systems. By accurately predicting ridership based on weather conditions, VTA authorities can make data-driven decisions to enhance service reliability, reduce delays, and promote sustainability.

## References
- VTA Open Data - Ridership Data: [Link](https://data.vta.org/documents/a3e899b0822e433fb52dda8a2d4f140c/about)
- NOAA Weather and Station Data: [Link](https://noaa-ghcn-pds.s3.amazonaws.com/index.html#csv/)
