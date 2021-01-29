#! /usr/bin/env python3
import pandas as pd
from math import sqrt
from matplotlib import cm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# keep to 3d
from mpl_toolkits.mplot3d import Axes3D

def calculate_wcss(data, max_grupos):
    wcss = []

    for n in range(2, max_grupos):
        kmeans = KMeans(n_clusters=n, random_state=10)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)

    return wcss


def optimal_number_of_clusters(wcss, max_grupos):
    x1, y1 = 2, wcss[0]
    x2, y2 = max_grupos - 1, wcss[len(wcss) - 1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2



def silhouette(dataset, n):

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    if n < 5:
        range_n_clusters = [2, 3, 4, 5, 6, 7]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataset) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(dataset)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataset, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(dataset[:, 0], dataset[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


def run_clustering(dataset, dataset_name, dataset_idx):
    pca = PCA(n_components=3)
    dataset_pca = pca.fit(dataset).transform(dataset)

    max_grupos = 30
    if len(dataset) < max_grupos:
        max_grupos = len(dataset)

    # calculando a soma dos quadrados para a quantidade de clusters
    sum_of_squares = calculate_wcss(dataset, max_grupos)

    # calculando a quantidade ótima de clusters
    n = optimal_number_of_clusters(sum_of_squares, max_grupos)

    # grafico 1
    fig = plt.figure(figsize=[10, 7])
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2])
    plt.title('%s ' % dataset_name)

    # grafico 2
    ax = fig.add_subplot(222)
    ax.plot(range(2, max_grupos), sum_of_squares, 'b*-')
    ax.plot(n, sum_of_squares[n - 2], marker='o', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Número de Grupos')
    plt.ylabel('Média soma dos quadrados intra-grupo')
    plt.title('Cotovelo KMeans')

    print("--- %s - número ideal de grupos --> %s" % (dataset_name, n))

    kmeans = KMeans(n_clusters=n,random_state=10)
    kmeans.fit(dataset)

    # grafico 3
    ax = fig.add_subplot(223, projection='3d')
    labels_kmeans = kmeans.labels_

    print("---KMeans silhouette -->", silhouette_score(dataset, labels_kmeans))

    ax.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    result = {}
    for i in range(0, len(dataset)):
        ax.text(dataset_pca[i - 1, 0], dataset_pca[i - 1, 1], dataset_pca[i - 1, 2], dataset_idx[i])
        result[dataset_idx[i]] = labels_kmeans[i - 1]
    plt.title('Textos - por índice')

    # grafico 4
    ax = fig.add_subplot(224, projection='3d')
    ax.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2],
               c=labels_kmeans.astype(np.float), edgecolor='k')

    for i in range(0, len(dataset)):
        ax.text(dataset_pca[i - 1, 0], dataset_pca[i - 1, 1], dataset_pca[i - 1, 2], labels_kmeans[i - 1])

    plt.title('Grupos')
    plt.show()

    print("kmeans - %s" % dataset_name, result)

    # ----------------------- AgglomerativeClustering - Hierarchical -------------------------
    agglomer = AgglomerativeClustering(n_clusters=n)
    agglomer.fit(dataset)
    labels_agglomer = agglomer.labels_

    print("---Agglomer silhouette -->", silhouette_score(dataset, labels_agglomer))

    result = {}
    for i in range(0, len(dataset)):
        result[dataset_idx[i]] = labels_agglomer[i - 1]

    print("agglomerative - %s " % (dataset_name), result)

    homo, comp, v_m = homogeneity_completeness_v_measure(labels_kmeans, labels_agglomer)
    print('Homogeneity (kmeans vs agglomer): {:.2f}%'.format(homo * 100))
    print('Completeness (kmeans vs agglomer): {:.2f}%'.format(comp * 100))
    print('V-Measure (kmeans vs agglomer): {:.2f}%'.format(v_m * 100))



def main():
    df = pd.read_csv('https://raw.githubusercontent.com/sidleal/jdp2019/master/rastros151.csv', index_col=0)

    # ----- Features ------
    # complexidade estrutural do período (períodos simples vs. compostos)
    estruturais = ['words_per_sentence', 'sentences', 'words', 'sentence_length_max', 'sentence_length_min',
                   'sentence_length_standard_deviation', 'yngve', 'frazier', 'words_before_main_verb',
                   'clauses_per_sentence', 'sentences_with_five_clauses', 'sentences_with_four_clauses',
                   'sentences_with_seven_more_clauses', 'sentences_with_six_clauses', 'sentences_with_three_clauses',
                   'punctuation_diversity', 'punctuation_ratio', 'non_svo_ratio', 'sentences_with_one_clause',
                   'sentences_with_two_clauses', 'sentences_with_zero_clause']
    # tipos de sentenças (ativas/passivas/relativas)
    tipos_sentenca = ['passive_ratio', 'relative_clauses', 'relative_pronouns_diversity_ratio', 'subordinate_clauses',
                      'infinite_subordinate_clauses', 'coordinate_conjunctions_per_clauses', 'apposition_per_clause']

    # mecanismos de construção de relações de correferência, entre outros
    correferencia = ['adjacent_refs', 'anaphoric_refs', 'adj_arg_ovl', 'arg_ovl', 'adj_stem_ovl', 'stem_ovl',
                     'adj_cw_ovl', 'coreference_pronoum_ratio']

    # informações sobre as categorias gramaticais: adjectives, adverbs, conjunctions, determiners, nouns, prepositions,
    # pronouns,  verbs; e flexão dos substantivos e verbos.
    morfossintaticas = ['adjective_ratio', 'adverbs', 'noun_ratio', 'verbs', 'pronoun_ratio',
                        'adjectives_standard_deviation', 'adverbs_max', 'adverbs_min', 'adverbs_standard_deviation',
                        'verbal_time_moods_diversity', 'nouns_max', 'nouns_min', 'nouns_standard_deviation',
                        'preposition_diversity', 'pronouns_max', 'pronouns_min', 'pronouns_standard_deviation',
                        'verbs_max', 'verbs_min', 'verbs_standard_deviation', 'syllables_per_content_word']

    all_sel = estruturais + tipos_sentenca + correferencia + morfossintaticas

    X_r = df[all_sel]
    # X_r = df[estruturais]

    # ----- PCA ------
    pca = PCA(n_components=3)
    X_pca = pca.fit(X_r).transform(X_r)

    # ----- Gráfico 1 - visão geral ------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
    plt.title('')
    plt.show()

    # ----- Gêneros ------
    divulgacao = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
                  37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
    literario = [122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,
                 147,148,149,150,151]
    jornalistico = [61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,
                    93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
                    119,120,121]

    #após rodar a primeira vez, pode-se pegar os grupos para plotar novamente e ver a diversidade
    selecao = [3, 9, 11, 12, 14, 16, 18, 21, 24, 25, 27, 31, 34, 37, 41, 42, 46, 47, 58, 60, 62, 71, 76, 83, 85, 90,
               94, 98, 99, 100, 101, 102, 105, 110, 111, 112, 113, 117, 118, 120, 122, 132, 133, 136, 138, 140, 143,
               144, 146, 149]

    X_g = [
        ["Gênero Jornalístico", jornalistico],
        ["Gênero Literário", literario],
        ["Gênero Divulgação Científica", divulgacao],
        ["Seleção", selecao]
    ]

    # ----- Execução ------
    for genre in X_g:
        print(genre[0])

        subset = X_r.loc[genre[1]]

        X_n = subset
        X_n = pca.fit(X_n).transform(X_n)

        run_clustering(X_n, genre[0], genre[1])


if __name__ == "__main__":
    main()
