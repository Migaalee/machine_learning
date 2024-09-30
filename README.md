# machine_learning
## Tutorials_AA_Migla.ipynb

I put together this tutorial to guide through the ins and outs of handling data for machine learning. Here I cover everything from feature selection and extraction to playing around with regression models and clustering algorithms. Datasets keep things interesting, whether it is the classic iris flower measurements or even some veggies in image form (yes, vegetables can teach you machine learning too).

The exercises are designed to walk through important concepts like normalizing your data (because nobody likes messy data), imputing missing values (because life is not perfect, and neither is data), and expanding features (because sometimes, bigger is better). By the end of this, anyone can train simple machine learning models like a pro.

-- poly_16features(X): Expands data features polynomially, creating interaction terms to fit more complex models.

-- poly_mat(reg, X_data, feats, ax_lims): Creates a score matrix for contour plots, visualizing decision boundaries of classifiers.

-- create_plot(X_r, Y_r, X_t, Y_t, feats, best_c): Plots the decision boundaries and classified points for logistic regression models.

-- calc_fold(feats, X, Y, train_ix, test_ix, C): Calculates the Brier score (mean squared difference between predicted probability and actual label) for each fold of a logistic regression classifier.

-- mink_dist(x, X, p): Computes Minkowski distance (with Euclidean as default) between a point x and all points in X for k-nearest neighbors (k-NN) classification.

-- knn_classify(x, X, Y, k): Classifies a point x using the k-nearest neighbors method, assigning it the most common label in the nearest neighbors.

-- nad_wat(K, h, X, Y, x): Implements the Nadaraya-Watson kernel regression, which is a non-parametric method for estimating the relationship between X and Y.

-- load_planet_data(file_name): Reads the planet dataset and returns the orbital radius and period, which is used for regression analysis.

-- mean_square_error(data, coefs): Calculates the mean squared error between predicted and actual values for regression tasks.

-- plot_svm(data, sv, f_name, C): Visualizes the decision boundaries of a Support Vector Machine (SVM) classifier for different regularization values (C).

-- plot_iris(X, y, file_name): Plots the Iris dataset with three different species in a 2D scatter plot.

-- plot_3d_color_space(cols): Plots color pixels in 3D RGB space, used to visualize image quantization using K-Means clustering.


## Naive_Bayes_Gaussian_Bayes_SVM.ipynb

In this project, we tackle the challenge of authenticating banknotes using machine learning models. The dataset is inspired by the UCI banknote authentication problem but has been adapted for this assignment. Each record in the .tsv file represents a banknote, with features derived from Wavelet Transformed images (variance, skewness, kurtosis, and entropy). The goal is to classify each banknote as either real (0) or fake (1).

Plot Training and Validation Errors: Visualize how  model performs on training and validation data for different bandwidths. Think of it like charting the performance of your model—a clear picture makes all the difference.

Find the Best Bandwidth: optimize the bandwidth for the Naive Bayes classifier to minimize validation error. It’s like tuning a radio station—find the sweet spot for the clearest signal.

True Error Calculation: We’ll measure the true error of the Naive Bayes classifier on the test set to see how it performs when faced with unseen data.

Gaussian Naive Bayes: We'll implement and evaluate a Naive Bayes classifier using Gaussian assumptions, to compare with the kernel-based version.

SVM Classifier: Use a Support Vector Machine with a Gaussian radial basis function (RBF) kernel. You’ll optimize the gamma parameter and see how that affects performance—because why settle for suboptimal settings?

Optimize Gamma Parameter: Try different gamma values and compare training and validation errors. The model’s performance is all about finding that perfect balance.

Plot Gamma Errors: Once the best gamma is found, we’ll visualize how it impacts training and validation errors, providing insights into model behavior.

Compare Classifiers: Finally, we’ll compare the performance of the Naive Bayes and SVM classifiers. It’s time for these models to face off—may the best one win (in terms of accuracy, of course)!


## DBSCAN_kmeans_bacteria_images.ipynb

### Bacterial Image Clustering Project
### Objective
The goal of this project is to examine a set of bacterial cell images using machine learning techniques such as feature extraction, feature selection, and clustering. The objective is to assist biologists in organizing similar images, providing insight into the best way to group these images.

### Dataset
The dataset consists of 563 images of bacteria, each sized 50x50 pixels with a black background and the segmented bacterial region centered. The task involves loading the images, extracting features, selecting the most relevant features, and applying clustering algorithms to group similar images.

### Key Steps:

Feature Extraction:
Three different dimensionality reduction techniques were used to extract features from the images:

Principal Component Analysis (PCA): Extracted six principal components that capture the maximum variance in the data.
t-Distributed Stochastic Neighbor Embedding (t-SNE): Reduced the dimensionality while maintaining local structure, producing six new features.
Isometric Mapping (Isomap): Preserved the geometry of the dataset while reducing dimensions, extracting six features.
Feature Selection:
After feature extraction, statistical methods such as ANOVA and SelectKBest were applied to select the best subset of features. These features were further refined using:

Mutual Information: Evaluated the dependency between the features and the target variable to remove redundant features.
Clustering Algorithms:
The project compares different clustering algorithms to find the most suitable one for grouping the bacterial images:

DBSCAN: A density-based clustering algorithm that finds clusters of varying shapes by identifying dense regions in the feature space.
K-Means: A partitioning algorithm that divides the dataset into a predefined number of clusters. The number of clusters was determined using methods like silhouette scores and the elbow method.
Clustering Performance Evaluation:
The performance of the clustering algorithms was assessed using various metrics:

Silhouette Score: Measures how well the clusters are separated.
Adjusted Rand Index: Evaluates how well the clustering results match the true labels.
Parameter Tuning:
The main parameters of each algorithm (e.g., epsilon for DBSCAN and the number of clusters for K-Means) were varied and evaluated to determine the optimal clustering settings.

Visualization:
Several visualization techniques were used to interpret the results:

Scatter Matrix and Correlation Matrix: Used to identify correlations between features.
Cluster Plots: Visualized the clusters formed by K-Means and DBSCAN.




