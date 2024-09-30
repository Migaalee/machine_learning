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
