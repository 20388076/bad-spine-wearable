import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------  Working Directory -----------------------------
# Set the working directory to the script's location if running in Visual Studio Code
import os
# Change working directory for this script
os.chdir(r'C:\Users\user\OneDrive\Έγγραφα\PlatformIO\Projects\bad-spine-wearable-1\src\data_process_and_classification') # modify this path to your working directory

# ============================= Utility Functions =============================

# ----------------------------- Import Libraries ------------------------------
import time
import sys
# ----------------------------- Kernel clean ----------------------------------
def cls():
    print(chr(27) + '[2J') 
# ----------------------------- Kernel pause ----------------------------------
def pause():
    input('PRESS ENTER TO CONTINUE.')
# ----------------------------- Process time count ----------------------------
def tic():
    return float(time.time())
# ----------------------------- Process time return ---------------------------
def toc(t1, s):
    t2 = float(time.time())
    print(f'{s} time taken: {t2 - t1:.6e} seconds')
# ----------------------------- Kernel break ----------------------------------
def RETURN():
    sys.exit()
# =============================================================================

# ----------------------------- Kernel clean call -----------------------------
cls()

# ---------------------------
# 1. Load Data
# ---------------------------
path = './4_FEATS_COMBINED/'
X = pd.read_csv(path + "all_raw_data.csv")
print("X shape:", X.shape)

# ---------------------------
# 2. Preprocessing
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: PCA for visualization (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------
# 3. Find optimal number of clusters (Elbow + Silhouette)
# ---------------------------
sil_scores = []
db_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    labels = kmeans.labels_
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, sil_scores, marker="o")
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")

plt.subplot(1, 2, 2)
plt.plot(K_range, db_scores, marker="o", color="red")
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Davies-Bouldin Index (lower = better)")
plt.show()

# Choose best k (max silhouette, min DB index)
best_k = K_range[np.argmax(sil_scores)]
print(f"Best k suggested by Silhouette: {best_k}")

# ---------------------------
# 4. Run clustering with best_k
# ---------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

agg = AgglomerativeClustering(n_clusters=best_k)
y_agg = agg.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=10)
y_dbscan = dbscan.fit_predict(X_scaled)

# ---------------------------
# 5. Save generated labels
# ---------------------------
pd.DataFrame(y_kmeans, columns=["label"]).to_csv("y_data_kmeans.csv", index=False)
pd.DataFrame(y_agg, columns=["label"]).to_csv("y_data_agg.csv", index=False)
pd.DataFrame(y_dbscan, columns=["label"]).to_csv("y_data_dbscan.csv", index=False)
print("Cluster labels saved to CSV files!")

# ---------------------------
# 6. Visualization (2D PCA)
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap="tab10", s=10)
axes[0].set_title(f"KMeans (k={best_k})")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_agg, cmap="tab10", s=10)
axes[1].set_title(f"Agglomerative (k={best_k})")

axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap="tab10", s=10)
axes[2].set_title("DBSCAN")

plt.suptitle("Clustering Results (PCA-reduced space)")
plt.show()
