# pipeline_mk.py
# интеграционный модуль: собирает все этапы обработки македонского текста в один pipeline
# вход: сырой текст на македонском
# выход: словарь с предложениями, графом, метриками и путём к визуализации

import sys
import os
import time
import logging

# принудительно ставим UTF-8, чтобы кириллица не ломалась в Windows-консоли
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# путь к корню проекта (на один уровень выше папки source/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# папка с переиспользуемыми модулями (Graph, Vertex, Edge, visualize)
TREE_DIR = os.path.join(PROJECT_ROOT, 'rulebased-concept-tree-main')
# добавляем в sys.path, чтобы можно было импортировать graph.graph и sent_class
if TREE_DIR not in sys.path:
    sys.path.insert(0, TREE_DIR)

# папка source (для импорта prepare_mk, make_graph_mk, metrics_mk, mk_rb_anaphora)
SOURCE_DIR = os.path.join(PROJECT_ROOT, 'source')
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# импортируем модуль подготовки текста (токенизация, POS, лемматизация, dependency parsing)
from prepare_mk import process_text_mk

# импортируем модуль построения семантического графа
from make_graph_mk import build_mk_graph

# импортируем модуль разрешения анафоры
from mk_rb_anaphora import resolve_anaphora_mk

# импортируем модуль метрик (графовые, текстовые, юмора)
from metrics_mk import (
    calculate_metrics_mk,
    calculate_ttr_mk,
    calculate_mtld_mk,
    average_word_length_mk,
    average_syllables_per_word_mk,
    calculate_t_score_mk,
    semantic_incongruity_score,
    lexical_surprise_mk,
)

# импортируем функцию интерактивной визуализации графа (HTML через vis.js)
from graph.graph import visualize_graph_interactive

# настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
# логгер для нашего модуля
logger = logging.getLogger('pipeline_mk')


def run_pipeline_mk(text, use_anaphora=True, visualize=True, output_dir="temp/visualization"):
    """
    Главная функция: прогоняет текст через весь pipeline и возвращает словарь с результатами.

    Параметры:
        text          -- сырой текст на македонском языке
        use_anaphora  -- нужно ли разрешать анафору (True/False)
        visualize     -- нужно ли сохранять HTML-визуализацию графа (True/False)
        output_dir    -- папка для сохранения визуализации (по умолчанию temp/visualization)

    Возвращает словарь:
        sentences, graph, graph_metrics, text_metrics, humor_metrics, visualization_path
    """
    # засекаем общее время работы pipeline
    pipeline_start = time.time()

    # словарь, куда будем складывать результаты каждого этапа
    results = {
        'sentences': None,
        'graph': None,
        'graph_metrics': None,
        'text_metrics': None,
        'humor_metrics': None,
        'visualization_path': None,
    }

    # --- этап 1: обработка текста (CLASSLA + spaCy) ---
    logger.info("этап 1/8: обработка текста через prepare_mk...")
    try:
        # process_text_mk разбивает текст на предложения с морфо-синтаксическим разбором
        sentences = process_text_mk(text)
        # сохраняем результат
        results['sentences'] = sentences
        logger.info(f"  получено {len(sentences)} предложений")
    except Exception as e:
        # если подготовка текста упала, дальнейшие этапы невозможны
        logger.error(f"  ошибка на этапе подготовки текста: {e}")
        return results

    # --- этап 2 (опциональный): разрешение анафоры ---
    if use_anaphora:
        logger.info("этап 2/8: разрешение анафоры через mk_rb_anaphora...")
        try:
            # resolve_anaphora_mk заменяет местоимения на их антецеденты
            sentences = resolve_anaphora_mk(sentences)
            # обновляем результат (sentences мог измениться in-place, но на всякий случай перезаписываем)
            results['sentences'] = sentences
            logger.info("  анафора разрешена")
        except Exception as e:
            # если анафора упала, продолжаем с исходными предложениями
            logger.warning(f"  ошибка при разрешении анафоры, пропускаем: {e}")
    else:
        logger.info("этап 2/8: разрешение анафоры пропущено (use_anaphora=False)")

    # --- этап 3: построение семантического графа ---
    logger.info("этап 3/8: построение семантического графа через make_graph_mk...")
    try:
        # build_mk_graph строит граф: вершины = именные группы, ребра = отношения
        graph = build_mk_graph(sentences)
        # сохраняем результат
        results['graph'] = graph
        # считаем количество вершин и ребер для отчёта
        n_verts = len(graph.vertices) if graph.vertices else 0
        n_edges = len(graph.edges) if graph.edges else 0
        logger.info(f"  граф построен: {n_verts} вершин, {n_edges} ребер")
    except Exception as e:
        # если граф не построился, метрики графа и визуализация будут недоступны
        logger.error(f"  ошибка при построении графа: {e}")

    # --- этап 4: метрики графа ---
    if results['graph'] is not None:
        logger.info("этап 4/8: вычисление графовых метрик через metrics_mk...")
        try:
            # calculate_metrics_mk считает 16 структурных метрик (степени, кластеризация и т.д.)
            graph_metrics = calculate_metrics_mk(results['graph'])
            # сохраняем результат
            results['graph_metrics'] = graph_metrics
            logger.info(f"  вычислено {len(graph_metrics)} графовых метрик")
        except Exception as e:
            logger.warning(f"  ошибка при вычислении графовых метрик: {e}")
    else:
        logger.info("этап 4/8: пропущен (граф не построен)")

    # --- этап 5: текстовые метрики ---
    logger.info("этап 5/8: вычисление текстовых метрик...")
    # словарь для текстовых метрик (TTR, MTLD, средняя длина слова, слоги, T-score)
    text_metrics = {}

    # TTR -- Type-Token Ratio (лексическое разнообразие)
    try:
        text_metrics['ttr'] = calculate_ttr_mk(text)
        logger.info(f"  TTR = {text_metrics['ttr']:.4f}")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении TTR: {e}")
        text_metrics['ttr'] = None

    # MTLD -- Measure of Textual Lexical Diversity (устойчив к длине текста)
    try:
        text_metrics['mtld'] = calculate_mtld_mk(text)
        logger.info(f"  MTLD = {text_metrics['mtld']:.4f}")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении MTLD: {e}")
        text_metrics['mtld'] = None

    # средняя длина слова в символах
    try:
        text_metrics['avg_word_length'] = average_word_length_mk(text)
        logger.info(f"  средняя длина слова = {text_metrics['avg_word_length']:.4f}")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении средней длины слова: {e}")
        text_metrics['avg_word_length'] = None

    # среднее количество слогов на слово
    try:
        text_metrics['avg_syllables_per_word'] = average_syllables_per_word_mk(text)
        logger.info(f"  средние слоги на слово = {text_metrics['avg_syllables_per_word']:.4f}")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении средних слогов: {e}")
        text_metrics['avg_syllables_per_word'] = None

    # T-score для биграмм (пар соседних лемм)
    try:
        text_metrics['t_score'] = calculate_t_score_mk(text)
        # количество биграмм с ненулевым T-score
        n_bigrams = len(text_metrics['t_score']) if text_metrics['t_score'] else 0
        logger.info(f"  T-score: {n_bigrams} биграмм")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении T-score: {e}")
        text_metrics['t_score'] = None

    # сохраняем все текстовые метрики
    results['text_metrics'] = text_metrics

    # --- этап 6: метрики юмора ---
    logger.info("этап 6/8: вычисление метрик юмора...")
    # словарь для метрик юмора
    humor_metrics = {}

    # semantic incongruity -- средняя семантическая неконгруэнтность ребер графа
    if results['graph'] is not None:
        try:
            humor_metrics['semantic_incongruity'] = semantic_incongruity_score(results['graph'])
            logger.info(f"  semantic incongruity = {humor_metrics['semantic_incongruity']:.4f}")
        except Exception as e:
            logger.warning(f"  ошибка при вычислении semantic incongruity: {e}")
            humor_metrics['semantic_incongruity'] = None
    else:
        # граф не построен, semantic incongruity вычислить нельзя
        humor_metrics['semantic_incongruity'] = None

    # lexical surprise -- средняя лексическая неожиданность текста
    try:
        humor_metrics['lexical_surprise'] = lexical_surprise_mk(text)
        logger.info(f"  lexical surprise = {humor_metrics['lexical_surprise']:.4f}")
    except Exception as e:
        logger.warning(f"  ошибка при вычислении lexical surprise: {e}")
        humor_metrics['lexical_surprise'] = None

    # сохраняем метрики юмора
    results['humor_metrics'] = humor_metrics

    # --- этап 7: визуализация графа ---
    if visualize and results['graph'] is not None:
        logger.info("этап 7/8: визуализация графа (интерактивный HTML)...")
        try:
            # абсолютный путь к папке для визуализации
            abs_output_dir = os.path.join(PROJECT_ROOT, output_dir)
            # создаём папку, если её ещё нет
            os.makedirs(abs_output_dir, exist_ok=True)
            # путь к HTML-файлу с визуализацией
            output_path = os.path.join(abs_output_dir, 'pipeline_graph.html')
            # visualize_graph_interactive сохраняет интерактивный HTML через vis.js
            visualize_graph_interactive(results['graph'], output=output_path)
            # сохраняем путь к файлу
            results['visualization_path'] = output_path
            logger.info(f"  граф сохранён: {output_path}")
        except Exception as e:
            logger.warning(f"  ошибка при визуализации графа: {e}")
    else:
        # визуализация отключена или граф не построен
        reason = "visualize=False" if not visualize else "граф не построен"
        logger.info(f"этап 7/8: визуализация пропущена ({reason})")

    # --- этап 8: финальный отчёт ---
    # считаем общее время работы pipeline
    elapsed = time.time() - pipeline_start
    logger.info(f"этап 8/8: pipeline завершён за {elapsed:.1f} секунд")

    return results


def print_results(results):
    """
    Красиво выводит результаты pipeline в консоль.
    Принимает словарь, который вернула run_pipeline_mk().
    """
    print()
    print("=" * 60)
    print("  РЕЗУЛЬТАТЫ PIPELINE")
    print("=" * 60)

    # --- предложения ---
    sentences = results.get('sentences')
    if sentences is not None:
        print(f"\nПредложений: {len(sentences)}")
    else:
        print("\nПредложения: не получены (ошибка)")

    # --- граф ---
    graph = results.get('graph')
    if graph is not None:
        n_verts = len(graph.vertices) if graph.vertices else 0
        n_edges = len(graph.edges) if graph.edges else 0
        print(f"\nГраф: {n_verts} вершин, {n_edges} ребер")

        # выводим первые 5 ребер как пример
        if graph.edges:
            print("\nПримеры ребер (первые 5):")
            for edge in graph.edges[:5]:
                print(f"  {edge.agent_1} --[{edge.meaning}]--> {edge.agent_2}")
    else:
        print("\nГраф: не построен (ошибка)")

    # --- графовые метрики ---
    graph_metrics = results.get('graph_metrics')
    if graph_metrics is not None:
        print(f"\nГрафовые метрики ({len(graph_metrics)} штук):")
        for key, value in graph_metrics.items():
            # форматируем числа с 4 знаками после запятой
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("\nГрафовые метрики: не вычислены")

    # --- текстовые метрики ---
    text_metrics = results.get('text_metrics')
    if text_metrics is not None:
        print("\nТекстовые метрики:")
        # TTR
        ttr = text_metrics.get('ttr')
        if ttr is not None:
            print(f"  TTR (Type-Token Ratio): {ttr:.4f}")
        else:
            print("  TTR: ошибка")

        # MTLD
        mtld = text_metrics.get('mtld')
        if mtld is not None:
            print(f"  MTLD (Measure of Textual Lexical Diversity): {mtld:.4f}")
        else:
            print("  MTLD: ошибка")

        # средняя длина слова
        awl = text_metrics.get('avg_word_length')
        if awl is not None:
            print(f"  средняя длина слова: {awl:.4f}")
        else:
            print("  средняя длина слова: ошибка")

        # средние слоги на слово
        aspw = text_metrics.get('avg_syllables_per_word')
        if aspw is not None:
            print(f"  средние слоги на слово: {aspw:.4f}")
        else:
            print("  средние слоги на слово: ошибка")

        # T-score (показываем top-5 биграмм)
        t_scores = text_metrics.get('t_score')
        if t_scores is not None and len(t_scores) > 0:
            # сортируем по T-score убыванию
            sorted_bigrams = sorted(t_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  T-score (top-5 из {len(sorted_bigrams)} биграмм):")
            for (w1, w2), score in sorted_bigrams[:5]:
                print(f"    ({w1}, {w2}): {score:.4f}")
        elif t_scores is not None:
            print("\n  T-score: биграмм не найдено")
        else:
            print("\n  T-score: ошибка")
    else:
        print("\nТекстовые метрики: не вычислены")

    # --- метрики юмора ---
    humor_metrics = results.get('humor_metrics')
    if humor_metrics is not None:
        print("\nМетрики юмора:")
        # semantic incongruity
        si = humor_metrics.get('semantic_incongruity')
        if si is not None:
            print(f"  semantic incongruity: {si:.4f}")
        else:
            print("  semantic incongruity: не вычислена")

        # lexical surprise
        ls = humor_metrics.get('lexical_surprise')
        if ls is not None:
            print(f"  lexical surprise: {ls:.4f}")
        else:
            print("  lexical surprise: не вычислена")
    else:
        print("\nМетрики юмора: не вычислены")

    # --- визуализация ---
    viz_path = results.get('visualization_path')
    if viz_path is not None:
        print(f"\nВизуализация графа: {viz_path}")
    else:
        print("\nВизуализация графа: не создана")

    print()
    print("=" * 60)


if __name__ == "__main__":
    # демо-текст на македонском для тестирования pipeline
    # два коротких предложения из разных доменов (чтобы граф имел хотя бы пару вершин)
    demo_text = (
        "Професорот влезе во училницата и ги погледна студентите. "
        "Тие седеа тивко и читаа книги за историјата на Македонија."
    )

    print("Демо-текст:")
    print(demo_text)
    print()

    # запускаем pipeline с анафорой и визуализацией
    results = run_pipeline_mk(
        text=demo_text,
        use_anaphora=True,
        visualize=True,
        output_dir="temp/visualization",
    )

    # выводим результаты
    print_results(results)
