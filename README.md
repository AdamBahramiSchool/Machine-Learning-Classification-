# Machine-Learning-Classification-

# Colour Words

**Description:** The goal of this project is to predict color words based on RGB values. We use various machine learning techniques including Naïve Bayesian classifier, k-nearest neighbors, and decision tree classifiers.

**Steps:**

1. **Data Preparation:** The RGB values dataset is cleaned and preprocessed. This involves removing any inconsistencies in the data, dealing with missing values, and scaling the RGB values to ensure they are on a consistent scale.
2. **Model Creation:** We implement GaussianNB-based classifiers, and create versions for k-nearest neighbors and decision trees. For each model, we create versions that take RGB directly or convert to LAB/HVS color space. The conversion to LAB/HVS color space is done to potentially improve the performance of the models, as these color spaces may better represent how humans perceive color.
3. **Training and Evaluation:** The models are trained on the dataset and parameters are adjusted for optimal performance. The models are then evaluated on validation data and the scores are printed to understand the performance of each model.

# Case of the Unlabelled Weather

**Description:** The aim of this project is to predict the city origin of weather data based on various features including temperature, precipitation, snowfall, etc., using machine learning models such as random forest, kNN, and Bayesian classifiers.

**Steps:**

1. **Data Preprocessing:** The features are cleaned and normalized to ensure they are on a consistent scale. The labelled data is then split into training and validation sets.
2. **Model Training:** Machine learning models are trained on the labelled data to predict city names. The models learn to associate the weather features with the city names in the training data.
3. **Prediction:** The trained model is used to predict cities for unlabelled 2016 weather data. This involves feeding the unlabelled data into the model and outputting the predicted city names.

# Exploring the Weather

**Description:** The purpose of this project is to explore the structure of weather data using principal component analysis (PCA) and clustering techniques.

**Steps:**

1. **Data Preparation:** The weather data is cleaned and preprocessed. This involves removing any inconsistencies in the data, dealing with missing values, and normalizing the features to ensure they are on a consistent scale.
2. **PCA:** PCA is implemented to reduce the dimensionality of the data for visualization. This involves transforming the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
3. **Clustering:** KMeans clustering is used to identify clusters of similar weather observations. This involves partitioning the data into k clusters, where each observation belongs to the cluster with the nearest mean.
4. **Visualization:** The clusters obtained from PCA are plotted for visualization. A table is created showing the distribution of observations from each city in each cluster. This helps in understanding the structure and patterns in the weather data.
