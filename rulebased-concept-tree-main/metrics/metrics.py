from graph.graph import Graph
from typing import Dict
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import pymorphy3
import re
from nltk.stem import WordNetLemmatizer
import csv


def calculate_metrics(graph: Graph) -> Dict[str, float]:
    """
    Calculate various metrics for the given graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary containing the calculated metrics.
    """
    G = graph.convert_to_networkx()
    metrics = {
        # degree metrics
        "num_vertices": len(graph.vertices),
        "num_edges": len(graph.edges),
        "average_degree": (2 * len(graph.edges)) / len(graph.vertices) if graph.vertices else 0,
        "average_in_degree": sum(dict(G.in_degree()).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_out_degree": sum(dict(G.out_degree()).values()) / len(graph.vertices) if graph.vertices else 0,
        "density": (2 * len(graph.edges)) / (len(graph.vertices) * (len(graph.vertices) - 1)) if graph.vertices else 0,
        # other metrics
        "average_clustering_coefficient": nx.average_clustering(G) if graph.vertices else 0,
        "assortativity": nx.degree_assortativity_coefficient(G) if graph.vertices else 0,
        "connected_components": nx.number_connected_components(G) if graph.vertices else 0,
        "giant_component_size": len(max(nx.connected_components(G), key=len)) if graph.vertices else 0,
        # path metrics
        "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
        "diameter": nx.diameter(G) if nx.is_connected(G) else float('inf'),
        # centrality metrics
        "average_degree_centrality": sum(nx.degree_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_betweenness_centrality": sum(nx.betweenness_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_closeness_centrality": sum(nx.closeness_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
        "average_eigenvector_centrality": sum(nx.eigenvector_centrality(G).values()) / len(graph.vertices) if graph.vertices else 0,
    }

    return metrics


def distribution_of_degrees(graph: Graph, log_scale: bool = False) -> Dict[int, int]:
    """
    Calculate the distribution of vertex degrees in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    degree_distribution = {}
    
    for node in G.nodes():
        degree = G.degree(node)
        if degree not in degree_distribution:
            degree_distribution[degree] = 0
        degree_distribution[degree] += 1

    # Visualize the degree distribution
    if log_scale:
        plt.yscale("log")
    plt.bar(degree_distribution.keys(), degree_distribution.values())
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Distribution of Vertex Degrees")
    plt.show()

    return degree_distribution


def distribution_of_clustering_coefficients(graph: Graph, log_scale: bool = False) -> Dict[float, int]:
    """
    Calculate the distribution of clustering coefficients in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    clustering_distribution = {}
    
    for node in G.nodes():
        clustering_coeff = nx.clustering(G, node)
        if clustering_coeff not in clustering_distribution:
            clustering_distribution[clustering_coeff] = 0
        clustering_distribution[clustering_coeff] += 1

    # Visualize the clustering coefficient distribution
    if log_scale:
        plt.yscale("log")
    plt.bar(clustering_distribution.keys(), clustering_distribution.values())
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Count")
    plt.title("Distribution of Clustering Coefficients")
    plt.show()

    return clustering_distribution


def distribution_of_shortest_path_lengths(graph: Graph, log_scale: bool = False) -> Dict[int, int]:
    """
    Calculate the distribution of shortest path lengths in the graph and visualize it.

    Args:
        graph: The Graph object to analyze
        log_scale: Whether to use a logarithmic scale for the y-axis
    """
    G = graph.convert_to_networkx()
    path_length_distribution = {}
    
    if nx.is_connected(G):
        for source in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, source)
            for length in lengths.values():
                if length not in path_length_distribution:
                    path_length_distribution[length] = 0
                path_length_distribution[length] += 1

        # Visualize the shortest path length distribution
        if log_scale:
            plt.yscale("log")
        plt.bar(path_length_distribution.keys(), path_length_distribution.values())
        plt.xlabel("Shortest Path Length")
        plt.ylabel("Count")
        plt.title("Distribution of Shortest Path Lengths")
        plt.show()

    return path_length_distribution


def betweenness_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the betweenness centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their betweenness centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.betweenness_centrality(G)


def closeness_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the closeness centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their closeness centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.closeness_centrality(G)


def eigenvector_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the eigenvector centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their eigenvector centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.eigenvector_centrality(G)


def degree_centrality(graph: Graph) -> Dict[str, float]:
    """
    Calculate the degree centrality for each vertex in the graph.

    Args:
        graph: The Graph object to analyze

    Returns:
        A dictionary mapping vertex names to their degree centrality values.
    """
    G = graph.convert_to_networkx()
    return nx.degree_centrality(G)

# Classic non-graph metrics

# Prepairing
def get_tokens(text):
    return re.findall(r'\b[а-яёa-z]+\b', text.lower())

def get_lemmas(text, lang='ru') -> list:
    tokens = get_tokens(text)
    if lang == 'ru':
        morph = pymorphy3.MorphAnalyzer()
        return [morph.parse(w)[0].normal_form for w in tokens]
    elif lang == 'en':
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in tokens]
    
# Metrics
def average_word_length(text: str) -> float:
    """
    Calculate the average word length in the given corpus.
    Args:
        corpus: A string containing the text to analyze
    Returns:
        The average word length in the corpus.
    """
    words = get_tokens(text)
    if not words:
        return 0
    total_length = sum(len(word) for word in words)
    return total_length / len(words)


def calculate_ttr(text, lang='ru') -> float:
    lemmas = get_lemmas(text, lang)
    if not lemmas: 
        return 0
    return len(set(lemmas)) / len(lemmas)


def calculate_mtld(text, lang='ru', threshold=0.72):
    """Упрощенная реализация MTLD (расчет среднего фактора выживания лексем)"""
    lemmas = get_lemmas(text, lang)
    def count_factors(words):
        if not words: return 0
        factors = 0
        ttr_sum = 1.0
        types = set()
        tokens = 0
        for w in words:
            tokens += 1
            types.add(w)
            ttr = len(types) / tokens
            if ttr < threshold:
                factors += 1
                types = set()
                tokens = 0
        if tokens > 0:
            factors += (1 - ttr) / (1 - threshold)
        return len(words) / factors if factors > 0 else len(words)

    forward = count_factors(lemmas)
    backward = count_factors(lemmas[::-1])
    return (forward + backward) / 2


def average_word_length(text):
    words = get_tokens(text)
    if not words: return 0
    return sum(len(w) for w in words) / len(words)


def syllables_count(text, language='ru'):
    if language == 'ru':
        vowels = "аеёиоуыэюя"
    elif language == 'en':
        vowels = "aeiou"
    return len([char for char in text.lower() if char in vowels])


def average_syllables_per_word(text, language='ru'):
    words = get_tokens(text)
    if not words: return 0
    return syllables_count(text, language) / len(words)


with open('freqrnc2011.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        freq_dict = {row[0]: float(row[2]) for row in reader}


def calculate_t_score(text):
    lemmas = get_lemmas(text)
    bigrams_counts = {}
    unigrams_counts = {}
    N = len(lemmas)
    
    for i in range(len(lemmas) - 1):
        w1, w2 = lemmas[i], lemmas[i + 1]
        bigrams_counts[(w1, w2)] = bigrams_counts.get((w1, w2), 0) + 1
        unigrams_counts[w1] = unigrams_counts.get(w1, 0) + 1
    unigrams_counts[lemmas[-1]] = unigrams_counts.get(lemmas[-1], 0) + 1
    
    t_scores = {}
    for (w1, w2), obs in bigrams_counts.items():
        # expected  E = (f(w1) * f(w2)) / N
        exp = ((freq_dict.get(w1, 0.1) / (10 ** 6)) * (freq_dict.get(w2, 0.1) / (10 ** 6))) * N
        # Formula for T-score
        t_scores[(w1, w2)] = (obs - exp) / obs ** 0.5 if obs > 0 else 0
    return t_scores
