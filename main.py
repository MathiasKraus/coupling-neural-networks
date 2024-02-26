import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.stats as stats

from model import LLM_coupled

sns.set_palette("Paired")

n_samples = 200
n_features = 6
n_clusters = 3
cluster_proba = [0.2, 0.3, 0.5]
n_times = 100

plot_pca = False

# Insulin | BMI | Age | Gender | Carbohydrates | Sports Activity
cov_matrix = np.array([
    [1.0, 0.3, 0.1, 0.0, 0.4, -0.3],  # Insulin
    [0.3, 1.0, 0.2, 0.1, 0.2, -0.5],  # BMI
    [0.1, 0.2, 1.0, 0.0, 0.1, -0.2],  # Age
    [0.0, 0.1, 0.0, 1.0, 0.0,  0.0],  # Gender
    [0.4, 0.2, 0.1, 0.0, 1.0, -0.1],  # Carbohydrates
    [-0.3,-0.5,-0.2, 0.0,-0.1,  1.0]   # Sports Activity
])


mse = {'global': [],
       'local_oracle': [],
       'coupling_oracle': [],
       'local_kmeans_2': [],
       'coupling_kmeans_2': [],
       'local_kmeans_3': [],
       'coupling_kmeans_3': [],
       'local_kmeans_5': [],
       'coupling_kmeans_5': [],
       }

mae = {'global': [],
       'local_oracle': [],
       'coupling_oracle': [],
       'local_kmeans_2': [],
       'coupling_kmeans_2': [],
       'local_kmeans_3': [],
       'coupling_kmeans_3': [],
       'local_kmeans_5': [],
       'coupling_kmeans_5': [],
       }

for iteration in range(n_times):
    if iteration % 10:
        print(iteration)

    np.random.seed(iteration)

    # Define cluster assignments and their coefficients
    cluster_coefficients = np.random.uniform(0.5, 1.5, size=(n_clusters, n_features))
    cluster_means = np.random.uniform(-1.7, 1.7, size=(n_clusters, n_features))

    def gen_n(n):
        X = []
        y = np.zeros(n_samples)
        cluster_assignments = np.random.choice(np.arange(n_clusters), n, p=cluster_proba)
        # Iterate over patients and apply cluster-specific effects
        for i in range(n):
            cluster = cluster_assignments[i]
            coefficients = cluster_coefficients[cluster, :]

            means = cluster_means[cluster, :]
            
            # Sample correlated features for each patient and time step
            X_patient = np.random.multivariate_normal(means, cov_matrix)
            
            linear_effects = X_patient @ coefficients
            y[i] = linear_effects + np.random.normal(0, 0.1)
            X.append(X_patient)

        X = np.vstack(X)

        return X, y, cluster_assignments


    X, y, cluster_assignments = gen_n(n_samples)
    X_test, y_test, cluster_assignments_test = gen_n(n_samples)


    if plot_pca:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # Plot the reduced dataset in a scatter plot with cluster colors
        colors = sns.color_palette("colorblind", n_colors=3)

        for cluster, color in enumerate(colors):
            plt.scatter(X_reduced[cluster_assignments == cluster, 0], 
                        X_reduced[cluster_assignments == cluster, 1], 
                        c=color, 
                        label=f"Cluster {cluster}")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.title("PCA Reduced Dataset")
        plt.savefig('clusters_PCA.pdf')
        #plt.show()

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    y_pred_test = model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse['global'].append(mse_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mae['global'].append(mae_test)

    #print('Fully local linear model optimal clusters')

    pred_all = np.zeros(len(X))
    pred_all_test = np.zeros(len(X_test))
    for c in np.unique(cluster_assignments):
        X_cluster = X[cluster_assignments == c]
        y_cluster = y[cluster_assignments == c]

        model = LinearRegression()
        model.fit(X_cluster, y_cluster)
        y_pred = model.predict(X_cluster)
        pred_all[cluster_assignments == c] = y_pred

        X_cluster = X_test[cluster_assignments_test == c]
        y_cluster = y_test[cluster_assignments_test == c]
        y_pred = model.predict(X_cluster)

        pred_all_test[cluster_assignments_test == c] = y_pred

    mse_test = mean_squared_error(y_test, pred_all_test)
    mse['local_oracle'].append(mse_test)

    mae_test = mean_absolute_error(y_test, pred_all_test)
    mae['local_oracle'].append(mae_test)

    #print('Fully local linear model, non-optimal clusters')

    for k in [2,3,5]:
        kmeans = KMeans(n_clusters=k, random_state=0)

        kmeans.fit(X)

        pred_cluster = kmeans.predict(X)
        pred_cluster_test = kmeans.predict(X_test)

        pred_all = np.zeros(len(X))
        pred_all_test = np.zeros(len(X_test))
        for c in np.unique(pred_cluster):
            X_cluster = X[pred_cluster == c]
            y_cluster = y[pred_cluster == c]

            model = LinearRegression()
            model.fit(X_cluster, y_cluster)
            y_pred = model.predict(X_cluster)
            pred_all[pred_cluster == c] = y_pred

            X_cluster = X_test[pred_cluster_test == c]
            y_cluster = y_test[pred_cluster_test == c]
            y_pred = model.predict(X_cluster)

            pred_all_test[pred_cluster_test == c] = y_pred

        mse_test = mean_squared_error(y_test, pred_all_test)
        mse[f'local_kmeans_{k}'].append(mse_test)

        mae_test = mean_absolute_error(y_test, pred_all_test)
        mae[f'local_kmeans_{k}'].append(mae_test)

    #print('Coupled linear models optimal clusters')

    model = LLM_coupled(reg_strength=0.01, epochs=1000, verbose=False)
    model.fit(X, y, cluster_assignments)

    y_pred = model.predict(X, cluster_assignments)
    y_pred_test = model.predict(X_test, cluster_assignments_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse[f'coupling_oracle'].append(mse_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mae[f'coupling_oracle'].append(mae_test)


    for k in [2,3,5]:
        kmeans = KMeans(n_clusters=k, random_state=0)

        kmeans.fit(X)

        pred_cluster = kmeans.predict(X)
        pred_cluster_test = kmeans.predict(X_test)

        model = LLM_coupled(reg_strength=0.01, epochs=1000, verbose=False)
        model.fit(X, y, pred_cluster)

        y_pred = model.predict(X, pred_cluster)
        y_pred_test = model.predict(X_test, pred_cluster_test)

        mse_test = mean_squared_error(y_test, y_pred_test)
        mse[f'coupling_kmeans_{k}'].append(mse_test)

        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae[f'coupling_kmeans_{k}'].append(mae_test)

for k,v in mse.items():
    print(k)
    print(np.mean(v))


for k in mse.keys():
    for k2 in mse.keys():
        t_statistic, p_value = stats.ttest_ind(mse[k], mse[k2])
        print(f'{k} vs {k2}: {p_value}')