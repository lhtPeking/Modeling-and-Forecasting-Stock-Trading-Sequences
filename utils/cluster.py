from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
from networkx.algorithms import community
from dtaidistance import dtw
from sklearn.cluster import AgglomerativeClustering
from adjustText import adjust_text
import pandas as pd
import numpy as np

class BaseClustering:
    def __init__(self, df):
        self.df = df
        self.labels = None

    def fit(self):
        raise NotImplementedError

    def plot_pca(self, title="PCA Clustering Visualization"):
        pca = PCA(n_components=2)
        data_embedded = pca.fit_transform(self.df.values)

        plot_df = pd.DataFrame(data_embedded, columns=["PC1", "PC2"])
        plot_df['Cluster'] = self.labels
        plot_df['Label'] = self.df.index

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df['Cluster'], cmap='tab10', s=60)

        texts = []
        for _, row in plot_df.iterrows():
            texts.append(plt.text(row['PC1'], row['PC2'], row['Label'], fontsize=8, alpha=0.9))

        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        plt.title(title, fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class StaticFeatureKMeansClustering(BaseClustering):
    def __init__(self, df, n_clusters=5):
        super().__init__(df)
        self.n_clusters = n_clusters

    def fit(self):
        model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = model.fit_predict(self.df)
        return self.labels


class RollingCorrelationCommunityClustering:
    def __init__(self, price_df, window=60, corr_threshold=0.8):
        self.price_df = price_df
        self.window = window
        self.threshold = corr_threshold
        self.labels = None
        self.recent_corr = None

    def fit(self):
        returns = self.price_df.pct_change(fill_method=None).dropna()
        self.recent_corr = returns.iloc[-self.window:].corr()

        G = nx.Graph()
        for i in self.recent_corr.columns:
            for j in self.recent_corr.columns:
                if i != j and self.recent_corr.loc[i, j] > self.threshold:
                    G.add_edge(i, j, weight=self.recent_corr.loc[i, j])

        # 社区划分
        comms = community.label_propagation_communities(G)
        label_map = {}
        for i, group in enumerate(comms):
            for stock in group:
                label_map[stock] = i

        # 针对 price_df 中的所有股票生成标签
        tickers = list(self.price_df.columns)
        self.labels = [label_map.get(t, -1) for t in tickers]

        return self.labels

    def plot_graph(self):
        if self.recent_corr is None:
            raise ValueError("请先调用 .fit()")

        G = nx.Graph()
        for i in self.recent_corr.columns:
            for j in self.recent_corr.columns:
                if i != j and self.recent_corr.loc[i, j] > self.threshold:
                    G.add_edge(i, j, weight=self.recent_corr.loc[i, j])

        if G.number_of_edges() == 0:
            print("⚠️ 图中无足够强的边（相关性低于阈值），无法绘制")
            return

        valid_nodes = list(G.nodes)
        pos = nx.spring_layout(G, seed=42)

        # 提取参与节点的颜色标签
        node_colors = [self.labels[self.price_df.columns.get_loc(node)] for node in valid_nodes]

        plt.figure(figsize=(10, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            cmap='tab10',
            node_size=600,
            font_size=10,
            edge_color='gray'
        )
        plt.title("Rolling Correlation Community Clustering")
        plt.show()


class DTWClustering(BaseClustering):
    def __init__(self, time_series_dict, n_clusters=4):
        self.series_dict = time_series_dict
        self.n_clusters = n_clusters

    def fit(self):
        names = list(self.series_dict.keys())
        series = [np.array(self.series_dict[k]) for k in names]

        dist_matrix = np.zeros((len(series), len(series)))
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                dist = dtw.distance(series[i], series[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        model = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed', linkage='average')
        self.labels = model.fit_predict(dist_matrix)

        self.df = pd.DataFrame(dist_matrix, index=names, columns=names)
        return self.labels

    def plot_heatmap(self):
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df, cmap='Blues', annot=False)
        plt.title("DTW Distance Matrix Heatmap")
        plt.show()
