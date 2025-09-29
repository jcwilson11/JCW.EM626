import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from matplotlib.backends.backend_pdf import PdfPages

def purity_score(y_true, y_pred):
    total = len(y_true)
    purity_sum = 0
    for cluster_id in np.unique(y_pred):
        idx = np.where(y_pred == cluster_id)[0]
        if len(idx) == 0:
            continue
        true_labels = y_true[idx]
        most_common_count = np.bincount(true_labels).max()
        purity_sum += most_common_count
    return purity_sum / total

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    use_train_test_split = True
    random_state = 42

    if use_train_test_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=random_state
        )
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        dt_acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    else:
        clf = DecisionTreeClassifier(random_state=random_state)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        dt_acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])

    test_sample = np.array([[5.0, 3.4, 1.6, 0.2]])
    predicted_class_idx = clf.predict(test_sample)[0]
    predicted_class_name = target_names[predicted_class_idx]
    print("Decision Tree accuracy:", round(dt_acc, 3))
    print("Predicted class for [5.0, 3.4, 1.6, 0.2]:", predicted_class_name)

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    first10 = pd.DataFrame({
        "Index": np.arange(10),
        "True Species": [target_names[i] for i in y[:10]],
        "Cluster": cluster_labels[:10]
    })
    print("\nFirst 10 samples: True species vs K-Means cluster")
    print(first10.to_string(index=False))

    purity = purity_score(y, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    print("\nK-Means metrics:")
    print("Purity:", round(purity, 3))
    print("Adjusted Rand Index:", round(ari, 3))

    pdf_path = "iris_results.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.08, 0.95, "Iris Classification & Clustering", fontsize=18, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=False, rounded=True, fontsize=8)
        plt.title("Decision Tree")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=plt.gca(), values_format="d")
        plt.title("Decision Tree Confusion Matrix")
        pdf.savefig(fig)
        plt.close(fig)

        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
        fig = plt.figure(figsize=(8, 6))
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {cluster_id}")
        plt.legend()
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("K-Means Clusters (PCA-reduced)")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        table_text = "First 10 Samples: True Species vs K-Means Cluster\n\n"
        for _, row in first10.iterrows():
            table_text += f"Index {int(row['Index'])}: {row['True Species']} -> Cluster {int(row['Cluster'])}\n"
        fig.text(0.08, 0.95, "Cluster Assignments (First 10)", fontsize=16, va="top")
        fig.text(0.08, 0.88, table_text, fontsize=11, va="top", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF written to: {pdf_path}")

if __name__ == "__main__":
    main()
