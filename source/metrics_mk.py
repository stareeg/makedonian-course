# metrics_mk.py
# графовые и текстовые метрики для македонского языка
#
# 9.1 — метрики графа (степени, кластеризация, центральности, пути)
# 9.2 — regex-токенизатор для македонского
# 9.3 — лемматизация через CLASSLA
# 9.4 — подсчёт слогов по гласным + текстовые метрики (TTR, MTLD и др.)
# 9.5 — частотный словарь и T-score
# 9.6 — метрики юмора (semantic incongruity, lexical surprise)
# 9.7 — тестирование

import sys
import os
import re
import csv
import logging
import math
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib
# бэкенд Agg позволяет сохранять графики в файл без GUI-окна
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# добавляем корневую папку проекта в sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# папка с классами графа (higher_dim_graph.Graph, Edge, Vertex)
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'rulebased-concept-tree-main')
if GRAPH_DIR not in sys.path:
    sys.path.insert(0, GRAPH_DIR)

# папка source (для импорта prepare_mk)
SOURCE_DIR = os.path.join(PROJECT_ROOT, 'source')
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)


# конвертация нашего графа в nx.DiGraph

def _graph_to_nx(graph) -> nx.DiGraph:
    """
    Конвертирует объект graph.higher_dim_graph.Graph в networkx DiGraph.
    Метод convert_to_networkx() у Graph нет, поэтому делаем вручную.

    graph.vertices — Dict[str, Vertex], ключи это имена концептов
    graph.edges — List[Edge], у каждого Edge: agent_1, agent_2, meaning
    """
    # создаём пустой ориентированный граф
    G = nx.DiGraph()

    # добавляем все вершины (даже если у них нет рёбер)
    for concept in graph.vertices:
        G.add_node(concept)

    # добавляем рёбра с атрибутом meaning
    for edge in graph.edges:
        G.add_edge(edge.agent_1, edge.agent_2, meaning=edge.meaning)

    return G


# 9.1. Графовые метрики

def calculate_metrics_mk(graph) -> Dict[str, float]:
    """
    Вычисляет набор структурных метрик для семантического графа.
    Принимает объект higher_dim_graph.Graph (тот, что возвращает build_mk_graph).
    Возвращает словарь с 16 метриками.
    """
    # конвертируем наш граф в networkx DiGraph
    G = _graph_to_nx(graph)

    # количество вершин и рёбер
    n_vertices = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # если граф пустой, возвращаем нули
    if n_vertices == 0:
        return {k: 0 for k in [
            'num_vertices', 'num_edges', 'average_degree',
            'average_in_degree', 'average_out_degree', 'density',
            'average_clustering_coefficient', 'assortativity',
            'weakly_connected_components', 'giant_component_size',
            'average_shortest_path_length', 'diameter',
            'average_degree_centrality', 'average_betweenness_centrality',
            'average_closeness_centrality', 'average_eigenvector_centrality',
        ]}

    # средняя степень: общее число рёбер * 2 / вершины (для ориентированного графа in+out)
    average_degree = (2 * n_edges) / n_vertices

    # средняя входящая степень
    in_degrees = dict(G.in_degree())
    average_in_degree = sum(in_degrees.values()) / n_vertices

    # средняя исходящая степень
    out_degrees = dict(G.out_degree())
    average_out_degree = sum(out_degrees.values()) / n_vertices

    # плотность графа
    density = nx.density(G)

    # кластеризация: nx.average_clustering не работает на DiGraph напрямую,
    # переводим в неориентированный для корректного подсчёта
    avg_clustering = nx.average_clustering(G.to_undirected())

    # ассортативность по степени (работает на DiGraph)
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        # на маленьких графах может вернуть nan (деление на ноль в корреляции)
        if np.isnan(assortativity):
            assortativity = 0.0
    except (ValueError, ZeroDivisionError):
        # если все вершины имеют одинаковую степень, ассортативность не определена
        assortativity = 0.0

    # слабо связные компоненты (для DiGraph используем weakly_connected)
    wcc = list(nx.weakly_connected_components(G))
    n_wcc = len(wcc)

    # размер гигантской компоненты (наибольшей слабо связной)
    giant_component_size = max(len(c) for c in wcc)

    # метрики путей: средняя длина кратчайшего пути и диаметр
    # если граф не слабо связен, считаем по наибольшей компоненте
    if nx.is_weakly_connected(G):
        # для ориентированного графа считаем в неориентированной версии,
        # чтобы гарантировать связность для path-метрик
        G_undirected = G.to_undirected()
        avg_path_length = nx.average_shortest_path_length(G_undirected)
        diameter = nx.diameter(G_undirected)
    else:
        # берём подграф наибольшей слабо связной компоненты
        giant_nodes = max(wcc, key=len)
        G_giant = G.subgraph(giant_nodes).copy().to_undirected()
        avg_path_length = nx.average_shortest_path_length(G_giant)
        diameter = nx.diameter(G_giant)

    # центральности: degree
    degree_cent = nx.degree_centrality(G)
    avg_degree_centrality = sum(degree_cent.values()) / n_vertices

    # betweenness centrality
    betweenness_cent = nx.betweenness_centrality(G)
    avg_betweenness = sum(betweenness_cent.values()) / n_vertices

    # closeness centrality
    closeness_cent = nx.closeness_centrality(G)
    avg_closeness = sum(closeness_cent.values()) / n_vertices

    # eigenvector centrality: может не сойтись на разреженных графах
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
        avg_eigenvector = sum(eigenvector_cent.values()) / n_vertices
    except nx.PowerIterationFailedConvergence:
        # если метод не сошёлся, ставим 0
        avg_eigenvector = 0.0

    # собираем все метрики в словарь
    metrics = {
        'num_vertices': n_vertices,
        'num_edges': n_edges,
        'average_degree': average_degree,
        'average_in_degree': average_in_degree,
        'average_out_degree': average_out_degree,
        'density': density,
        'average_clustering_coefficient': avg_clustering,
        'assortativity': assortativity,
        'weakly_connected_components': n_wcc,
        'giant_component_size': giant_component_size,
        'average_shortest_path_length': avg_path_length,
        'diameter': diameter,
        'average_degree_centrality': avg_degree_centrality,
        'average_betweenness_centrality': avg_betweenness,
        'average_closeness_centrality': avg_closeness,
        'average_eigenvector_centrality': avg_eigenvector,
    }

    return metrics


# распределения (графовые)

def distribution_of_degrees(graph, save_path: Optional[str] = None) -> Dict[int, int]:
    """
    Строит распределение степеней вершин графа.
    Если save_path задан, сохраняет гистограмму в файл.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)

    # считаем сколько вершин имеет каждую степень
    degree_dist = {}
    for node in G.nodes():
        # суммарная степень (in + out)
        deg = G.degree(node)
        degree_dist[deg] = degree_dist.get(deg, 0) + 1

    # рисуем гистограмму
    plt.figure(figsize=(10, 6))
    plt.bar(degree_dist.keys(), degree_dist.values(), color='steelblue')
    plt.xlabel('Степен (degree)')
    plt.ylabel('Број на врвови (count)')
    plt.title('Распределба на степени на врвовите')
    plt.tight_layout()

    # сохраняем если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

    return degree_dist


def distribution_of_clustering_coefficients(graph, save_path: Optional[str] = None) -> Dict[float, int]:
    """
    Строит распределение коэффициентов кластеризации вершин.
    Кластеризацию считаем на неориентированной версии графа.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)

    # для кластеризации нужен неориентированный граф
    G_undirected = G.to_undirected()

    # считаем кластеризацию для каждой вершины
    clustering_dist = {}
    for node in G_undirected.nodes():
        cc = nx.clustering(G_undirected, node)
        # округляем до 2 знаков, чтобы группировать близкие значения
        cc_rounded = round(cc, 2)
        clustering_dist[cc_rounded] = clustering_dist.get(cc_rounded, 0) + 1

    # рисуем гистограмму
    plt.figure(figsize=(10, 6))
    # сортируем ключи для красивого графика
    sorted_keys = sorted(clustering_dist.keys())
    sorted_vals = [clustering_dist[k] for k in sorted_keys]
    plt.bar(sorted_keys, sorted_vals, width=0.02, color='darkorange')
    plt.xlabel('Коефициент на кластеризација')
    plt.ylabel('Број на врвови (count)')
    plt.title('Распределба на коефициенти на кластеризација')
    plt.tight_layout()

    # сохраняем если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

    return clustering_dist


def distribution_of_shortest_path_lengths(graph, save_path: Optional[str] = None) -> Dict[int, int]:
    """
    Строит распределение длин кратчайших путей.
    Если граф не связен, считаем по наибольшей слабо связной компоненте.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)

    # для кратчайших путей работаем с неориентированной версией
    if nx.is_weakly_connected(G):
        # граф связен, берём целиком
        G_paths = G.to_undirected()
    else:
        # берём наибольшую слабо связную компоненту
        giant_nodes = max(nx.weakly_connected_components(G), key=len)
        G_paths = G.subgraph(giant_nodes).copy().to_undirected()

    # считаем длины кратчайших путей для всех пар вершин
    path_dist = {}
    for source in G_paths.nodes():
        # BFS от source: возвращает словарь {target: length}
        lengths = nx.single_source_shortest_path_length(G_paths, source)
        for length in lengths.values():
            # пропускаем нулевые (расстояние до себя)
            if length == 0:
                continue
            path_dist[length] = path_dist.get(length, 0) + 1

    # рисуем гистограмму
    plt.figure(figsize=(10, 6))
    sorted_keys = sorted(path_dist.keys())
    sorted_vals = [path_dist[k] for k in sorted_keys]
    plt.bar(sorted_keys, sorted_vals, color='forestgreen')
    plt.xlabel('Должина на најкраток пат')
    plt.ylabel('Број на парови (count)')
    plt.title('Распределба на должини на најкратки патишта')
    plt.tight_layout()

    # сохраняем если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

    return path_dist


# индивидуальные центральности

def degree_centrality_mk(graph) -> Dict[str, float]:
    """
    Degree centrality для каждой вершины графа.
    Возвращает словарь {имя_концепта: значение}.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)
    return nx.degree_centrality(G)


def betweenness_centrality_mk(graph) -> Dict[str, float]:
    """
    Betweenness centrality для каждой вершины графа.
    Показывает, насколько вершина лежит на кратчайших путях между другими вершинами.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)
    return nx.betweenness_centrality(G)


def closeness_centrality_mk(graph) -> Dict[str, float]:
    """
    Closeness centrality для каждой вершины графа.
    Показывает, насколько близко вершина расположена к остальным.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)
    return nx.closeness_centrality(G)


def eigenvector_centrality_mk(graph) -> Dict[str, float]:
    """
    Eigenvector centrality для каждой вершины графа.
    Учитывает не только количество связей, но и важность соседей.
    Может не сойтись на разреженных графах, тогда возвращает пустой словарь.
    """
    # конвертируем в networkx
    G = _graph_to_nx(graph)

    try:
        # увеличиваем max_iter, чтобы дать алгоритму больше шансов сойтись
        return nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # если не сошёлся, возвращаем нули для всех вершин
        return {node: 0.0 for node in G.nodes()}


# 9.2. Токенизация для македонского

def get_tokens_mk(text: str) -> List[str]:
    """
    Regex-токенизатор для македонского языка.
    Выделяет слова, состоящие из букв кириллицы (включая специфичные
    македонские: ѓ, ѕ, ј, љ, њ, ќ, џ) и латиницы.
    Возвращает список токенов в нижнем регистре.
    """
    # приводим текст к нижнему регистру
    text_lower = text.lower()

    # regex покрывает русскую кириллицу (а-яё), македонские буквы и латиницу
    tokens = re.findall(r'\b[а-яёѓѕјљњќџa-z]+\b', text_lower, re.IGNORECASE)
    return tokens


# 9.3. Лемматизация через CLASSLA

def get_lemmas_mk(text: str) -> List[str]:
    """
    Лемматизатор для македонского через CLASSLA.
    Используем CLASSLA, а не pymorphy, потому что pymorphy не поддерживает
    македонский язык. CLASSLA (форк Stanza) обучен на южнославянских языках
    и качественно лемматизирует mk.

    Импортируем _get_classla_pipeline из prepare_mk, чтобы не создавать
    второй экземпляр pipeline (он тяжёлый и кешируется глобально).
    """
    # импортируем lazy-загрузчик pipeline из prepare_mk
    from prepare_mk import _get_classla_pipeline

    # получаем (или создаём при первом вызове) CLASSLA pipeline
    nlp = _get_classla_pipeline()

    # прогоняем текст через pipeline (tokenize + POS + lemma)
    doc = nlp(text)

    # собираем леммы из всех предложений и слов
    lemmas = []
    for sentence in doc.sentences:
        for word in sentence.words:
            # приводим лемму к нижнему регистру для единообразия
            lemmas.append(word.lemma.lower())

    return lemmas


# 9.4. Подсчёт слогов для македонского

def syllables_count_mk(text: str) -> int:
    """
    Считает количество слогов в тексте по числу гласных букв.
    В македонском всего 5 гласных: а, е, и, о, у.
    В отличие от русского, нет ё, ы, э, ю, я.
    """
    # македонские гласные
    vowels = "аеиоу"

    # приводим к нижнему регистру и считаем гласные
    count = sum(1 for char in text.lower() if char in vowels)
    return count


# текстовые метрики

def average_word_length_mk(text: str) -> float:
    """
    Средняя длина слова в тексте (в символах).
    """
    # токенизируем
    words = get_tokens_mk(text)

    # если текст пустой, возвращаем 0
    if not words:
        return 0.0

    # суммируем длины всех слов и делим на количество
    total_length = sum(len(w) for w in words)
    return total_length / len(words)


# TTR (Type-Token Ratio) — отношение уникальных лемм к общему числу лемм
def calculate_ttr_mk(text: str) -> float:
    """
    Вычисляет TTR: чем выше, тем разнообразнее лексика текста.
    """
    # получаем леммы через CLASSLA
    lemmas = get_lemmas_mk(text)

    # защита от пустого текста
    if not lemmas:
        return 0.0

    # TTR = количество уникальных лемм / общее количество лемм
    return len(set(lemmas)) / len(lemmas)


# MTLD (Measure of Textual Lexical Diversity) — устойчивая мера лексического разнообразия
def calculate_mtld_mk(text: str, threshold: float = 0.72) -> float:
    """
    Упрощённая реализация MTLD.
    В отличие от TTR, MTLD устойчив к длине текста:
    разбивает текст на отрезки, где TTR падает ниже threshold,
    и усредняет длину таких отрезков в прямом и обратном направлениях.
    """
    # получаем леммы через CLASSLA
    lemmas = get_lemmas_mk(text)

    def count_factors(words: List[str]) -> float:
        """
        Считает количество факторов (отрезков, где TTR упал ниже порога).
        """
        # если список пустой, возвращаем 0
        if not words:
            return 0.0

        # счётчик полных факторов
        factors = 0
        # множество уникальных слов в текущем отрезке
        types = set()
        # количество токенов в текущем отрезке
        tokens = 0

        for w in words:
            tokens += 1
            types.add(w)
            # текущий TTR для отрезка
            ttr = len(types) / tokens

            # если TTR упал ниже порога, отрезок закончен
            if ttr < threshold:
                factors += 1
                # начинаем новый отрезок с чистым множеством
                types = set()
                tokens = 0

        # неполный хвост: добавляем долю от полного фактора
        if tokens > 0:
            # текущий TTR на хвосте
            ttr = len(types) / tokens
            factors += (1 - ttr) / (1 - threshold)

        # MTLD = длина текста / количество факторов
        if factors > 0:
            return len(words) / factors
        # если факторов ноль (текст очень короткий), MTLD = длина текста
        return float(len(words))

    # считаем MTLD в прямом направлении
    forward = count_factors(lemmas)
    # считаем MTLD в обратном направлении
    backward = count_factors(lemmas[::-1])

    # итоговый MTLD — среднее двух направлений
    return (forward + backward) / 2


def average_syllables_per_word_mk(text: str) -> float:
    """
    Среднее количество слогов на слово в тексте.
    """
    # токенизируем
    words = get_tokens_mk(text)

    # защита от пустого текста
    if not words:
        return 0.0

    # общее число слогов / количество слов
    return syllables_count_mk(text) / len(words)


# 9.5. Частотный словарь и T-score

# логгер для предупреждений (например, если файл со словарём не найден)
logger = logging.getLogger(__name__)

# кеш для частотного словаря, чтобы не перечитывать файл каждый раз
_freq_dict_mk_cache: Optional[Dict[str, float]] = None


def build_frequency_dict_mk(
    texts_dir: str,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Строит частотный словарь по всему корпусу очищенных текстов.
    Считывает все .txt из texts_dir, токенизирует и считает частоты.
    Возвращает словарь {слово: частота_на_миллион}.
    """
    # счётчик абсолютных частот для каждого слова
    word_counts: Counter = Counter()

    # общее количество токенов во всём корпусе
    total_tokens = 0

    # проходим по всем файлам в папке с очищенными текстами
    for filename in os.listdir(texts_dir):
        # берём только .txt файлы
        if not filename.endswith('.txt'):
            continue

        # полный путь к файлу
        filepath = os.path.join(texts_dir, filename)

        # читаем содержимое файла
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # токенизируем текст нашим regex-токенизатором
        tokens = get_tokens_mk(text)

        # обновляем счётчик слов
        word_counts.update(tokens)

        # прибавляем к общему числу токенов
        total_tokens += len(tokens)

    # пустой корпус — возвращаем пустой словарь
    if total_tokens == 0:
        return {}

    # пересчитываем абсолютные частоты в частоту на миллион слов
    freq_per_million = {}
    for word, count in word_counts.items():
        freq_per_million[word] = (count / total_tokens) * 1_000_000

    # если указан путь для сохранения, записываем CSV
    if output_path is not None:
        # создаём папку, если её ещё нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # открываем файл для записи
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # заголовок таблицы
            writer.writerow(['word', 'count', 'frequency_per_million'])

            # сортируем по убыванию абсолютной частоты
            for word, count in word_counts.most_common():
                writer.writerow([word, count, freq_per_million[word]])

    return freq_per_million


def _load_freq_dict_mk() -> Dict[str, float]:
    """
    Загружает частотный словарь из CSV-файла datasets/freq_mk.csv.
    Результат кешируется в модульной переменной, чтобы не читать файл повторно.
    Если файл не найден, возвращает пустой словарь и выводит предупреждение.
    """
    global _freq_dict_mk_cache

    # если словарь уже загружен, просто отдаём из кеша
    if _freq_dict_mk_cache is not None:
        return _freq_dict_mk_cache

    # путь к файлу словаря
    freq_path = os.path.join(PROJECT_ROOT, 'datasets', 'freq_mk.csv')

    # проверяем, существует ли файл
    if not os.path.exists(freq_path):
        logger.warning(
            'частотный словарь %s не найден, T-score и surprise будут неточными',
            freq_path,
        )
        # кешируем пустой словарь, чтобы не проверять файл каждый раз
        _freq_dict_mk_cache = {}
        return _freq_dict_mk_cache

    # читаем CSV и заполняем словарь
    freq_dict: Dict[str, float] = {}
    with open(freq_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ключ — слово, значение — частота на миллион
            # в CSV столбец может называться freq_per_million или frequency_per_million
            freq_val = row.get('freq_per_million') or row.get('frequency_per_million')
            freq_dict[row['word']] = float(freq_val)

    # сохраняем в кеш
    _freq_dict_mk_cache = freq_dict
    return _freq_dict_mk_cache


def calculate_t_score_mk(text: str) -> Dict[Tuple[str, str], float]:
    """
    Вычисляет T-score для всех пар соседних лемм в тексте.
    T-score показывает, насколько часто два слова встречаются рядом
    по сравнению с тем, что мы ожидали бы случайно.

    Формула: (observed - expected) / sqrt(observed)
    expected = (freq(w1)/1M * freq(w2)/1M) * N
    """
    # получаем леммы текста через CLASSLA
    lemmas = get_lemmas_mk(text)

    # если лемм меньше двух, биграммы составить нельзя
    if len(lemmas) < 2:
        return {}

    # загружаем частотный словарь
    freq_dict = _load_freq_dict_mk()

    # общее количество лемм (N в формуле)
    n_lemmas = len(lemmas)

    # считаем абсолютные частоты биграмм (пар соседних лемм)
    bigram_counts: Counter = Counter()
    for i in range(len(lemmas) - 1):
        bigram = (lemmas[i], lemmas[i + 1])
        bigram_counts[bigram] += 1

    # считаем T-score для каждой биграммы
    t_scores: Dict[Tuple[str, str], float] = {}
    for (w1, w2), observed in bigram_counts.items():
        # частоты отдельных слов (на миллион); если слова нет в словаре, берём 0.01
        freq_w1 = freq_dict.get(w1, 0.01)
        freq_w2 = freq_dict.get(w2, 0.01)

        # ожидаемая частота биграммы при случайном соседстве
        # делим на 1M дважды, потому что freq уже в "на миллион"
        expected = (freq_w1 / 1_000_000) * (freq_w2 / 1_000_000) * n_lemmas

        # T-score: если observed == 0, пропускаем (но у нас observed >= 1)
        if observed > 0:
            t_score = (observed - expected) / math.sqrt(observed)
            t_scores[(w1, w2)] = t_score

    return t_scores


# 9.6. Метрики юмора

def semantic_incongruity_score(graph) -> float:
    """
    Считает среднюю семантическую неконгруэнтность графа.
    Для каждого ребра берём embedding вершин agent_1 и agent_2
    и считаем косинусное расстояние между ними.
    Чем выше значение, тем более неожиданные связи в графе (потенциально смешнее).
    """
    # список косинусных расстояний для каждого ребра
    distances = []

    for edge in graph.edges:
        # берём имена вершин на концах ребра
        name_1 = edge.agent_1
        name_2 = edge.agent_2

        # проверяем, что обе вершины есть в графе
        if name_1 not in graph.vertices or name_2 not in graph.vertices:
            continue

        # берём embedding каждой вершины (загружается автоматически в Vertex.__post_init__)
        emb_1 = graph.vertices[name_1].embedding
        emb_2 = graph.vertices[name_2].embedding

        # пропускаем, если у какой-то вершины нет embedding
        if emb_1 is None or emb_2 is None:
            continue

        # приводим к numpy-массивам на случай, если они ещё не numpy
        a = np.array(emb_1, dtype=np.float64)
        b = np.array(emb_2, dtype=np.float64)

        # считаем нормы векторов
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # если один из векторов нулевой, пропускаем (нельзя посчитать косинус)
        if norm_a == 0 or norm_b == 0:
            continue

        # косинусное сходство
        cosine_sim = np.dot(a, b) / (norm_a * norm_b)

        # косинусное расстояние = 1 - сходство (от 0 до 2)
        cosine_dist = 1.0 - cosine_sim

        distances.append(cosine_dist)

    # если ни одного ребра с embedding не нашлось, возвращаем 0
    if not distances:
        return 0.0

    # среднее расстояние по всем рёбрам
    return float(np.mean(distances))


def lexical_surprise_mk(text: str) -> float:
    """
    Считает среднюю лексическую неожиданность текста.
    Редкие слова дают больше "сюрприза", частые — меньше.

    Для каждого токена берём его частоту из словаря и считаем -log(freq).
    Низкая частота => большой -log => больше сюрприза.
    Слова, которых нет в словаре, получают минимальную частоту 0.01.
    """
    # токенизируем текст
    tokens = get_tokens_mk(text)

    # если текст пустой, возвращаем 0
    if not tokens:
        return 0.0

    # загружаем частотный словарь
    freq_dict = _load_freq_dict_mk()

    # минимальная частота для слов, которых нет в словаре
    default_freq = 0.01

    # считаем surprise = -log(freq) для каждого токена
    surprises = []
    for token in tokens:
        # частота на миллион; если слова нет, берём default
        freq = freq_dict.get(token, default_freq)

        # нормируем: делим на миллион, чтобы получить вероятность (от 0 до 1)
        prob = freq / 1_000_000

        # ограничиваем снизу, чтобы log не ушёл в бесконечность
        prob = max(prob, 1e-10)

        # surprise = -log(вероятность), чем реже слово, тем больше значение
        surprises.append(-math.log(prob))

    # возвращаем среднюю неожиданность по всему тексту
    return float(np.mean(surprises))


# 9.7. тестирование
if __name__ == '__main__':
    import time

    # принудительно ставим UTF-8, чтобы кириллица не ломалась в Windows-консоли
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # папка для сохранения результатов тестирования
    output_dir = os.path.join(PROJECT_ROOT, 'temp')
    os.makedirs(output_dir, exist_ok=True)

    # файл для записи всех результатов
    output_file = os.path.join(output_dir, 'phase_e_step9_output.txt')

    # список строк для записи в файл
    lines = []

    def log(msg: str):
        """Печатает строку в консоль и добавляет в список для записи."""
        print(msg)
        lines.append(msg)

    log('=' * 60)
    log('Тестирование metrics_mk.py (шаг 9.7)')
    log('=' * 60)

    # тестовые предложения на македонском
    test_sentences = [
        'Марко отиде на пазар и купи јаболка.',
        'Петар и Ана читаат книга во библиотеката.',
        'Кучето трча низ паркот и лае на мачката.',
    ]
    # объединяем в один текст для тестирования текстовых метрик
    test_text = ' '.join(test_sentences)

    # тестируем текстовые метрики (не требуют граф)
    log('\n--- Текстовые метрики ---')

    log(f'\nТестов текст ({len(test_text)} символи):')
    log(f'  "{test_text[:100]}..."')

    # 9.2: токенизация
    tokens = get_tokens_mk(test_text)
    log(f'\nget_tokens_mk: {len(tokens)} токени')
    log(f'  првите 15: {tokens[:15]}')

    # 9.4: подсчёт слогов
    n_syllables = syllables_count_mk(test_text)
    log(f'\nsyllables_count_mk: {n_syllables} слогови')

    # средняя длина слова
    avg_len = average_word_length_mk(test_text)
    log(f'average_word_length_mk: {avg_len:.2f} букви')

    # средние слоги на слово
    avg_syl = average_syllables_per_word_mk(test_text)
    log(f'average_syllables_per_word_mk: {avg_syl:.2f}')

    # 9.3: лемматизация через CLASSLA
    log('\nЗагрузка CLASSLA pipeline для лемматизации...')
    t0 = time.time()
    lemmas = get_lemmas_mk(test_text)
    t1 = time.time()
    log(f'get_lemmas_mk: {len(lemmas)} лемми (за {t1 - t0:.1f} сек)')
    log(f'  првите 15: {lemmas[:15]}')

    # TTR
    ttr = calculate_ttr_mk(test_text)
    log(f'\ncalculate_ttr_mk: {ttr:.4f}')

    # MTLD
    mtld = calculate_mtld_mk(test_text)
    log(f'calculate_mtld_mk: {mtld:.2f}')

    # 9.5: T-score
    log('\n--- T-score ---')
    t_scores = calculate_t_score_mk(test_text)
    log(f'calculate_t_score_mk: {len(t_scores)} биграми')
    # показываем top-5 по T-score
    sorted_ts = sorted(t_scores.items(), key=lambda x: x[1], reverse=True)
    for (w1, w2), score in sorted_ts[:5]:
        log(f'  ({w1}, {w2}): T-score = {score:.4f}')

    # 9.6: lexical surprise
    log('\n--- Lexical surprise ---')
    surprise = lexical_surprise_mk(test_text)
    log(f'lexical_surprise_mk: {surprise:.4f}')

    # тестируем графовые метрики
    log('\n--- Графовые метрики ---')
    log('Строим граф из тестовых предложений...')

    try:
        # импортируем функции построения графа
        from prepare_mk import process_text_mk
        from make_graph_mk import build_mk_graph

        # обрабатываем текст через NLP-pipeline
        t0 = time.time()
        sentences = process_text_mk(test_text)
        t1 = time.time()
        log(f'process_text_mk: {len(sentences)} предложенија (за {t1 - t0:.1f} сек)')

        # строим семантический граф
        t0 = time.time()
        graph = build_mk_graph(sentences)
        t1 = time.time()
        log(f'build_mk_graph: {len(graph.vertices)} врвови, {len(graph.edges)} ребра (за {t1 - t0:.1f} сек)')

        # проверяем, что граф не пустой
        if len(graph.vertices) > 0:
            # 9.1: основные метрики
            metrics = calculate_metrics_mk(graph)
            log(f'\ncalculate_metrics_mk: {len(metrics)} метрики')
            for name, value in metrics.items():
                if isinstance(value, float):
                    log(f'  {name}: {value:.4f}')
                else:
                    log(f'  {name}: {value}')

            # центральности
            log('\nIndividual centralities:')
            dc = degree_centrality_mk(graph)
            log(f'  degree_centrality: {len(dc)} вершини')
            bc = betweenness_centrality_mk(graph)
            log(f'  betweenness_centrality: {len(bc)} вершини')
            cc = closeness_centrality_mk(graph)
            log(f'  closeness_centrality: {len(cc)} вершини')
            ec = eigenvector_centrality_mk(graph)
            log(f'  eigenvector_centrality: {len(ec)} вершини')

            # распределения: сохраняем графики в temp/visualization/
            viz_dir = os.path.join(output_dir, 'visualization')
            os.makedirs(viz_dir, exist_ok=True)

            deg_dist = distribution_of_degrees(
                graph, save_path=os.path.join(viz_dir, 'degree_distribution.png'))
            log(f'\ndistribution_of_degrees: {deg_dist}')

            clust_dist = distribution_of_clustering_coefficients(
                graph, save_path=os.path.join(viz_dir, 'clustering_distribution.png'))
            log(f'distribution_of_clustering_coefficients: {clust_dist}')

            path_dist = distribution_of_shortest_path_lengths(
                graph, save_path=os.path.join(viz_dir, 'path_length_distribution.png'))
            log(f'distribution_of_shortest_path_lengths: {path_dist}')

            # 9.6: semantic incongruity
            log('\n--- Semantic incongruity ---')
            incongruity = semantic_incongruity_score(graph)
            log(f'semantic_incongruity_score: {incongruity:.4f}')
        else:
            log('Граф пустой, графовые метрики пропущены')

    except Exception as e:
        log(f'\nОшибка при построении графа: {type(e).__name__}: {e}')
        import traceback
        tb = traceback.format_exc()
        log(tb)

    # записываем все результаты в файл
    log(f'\n--- Результаты сохранены в {output_file} ---')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
