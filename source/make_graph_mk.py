# make_graph_mk.py
# модуль построения семантического графа для македонского языка
# принимает список объектов Sentence (из sent_class.py, после обработки через prepare_mk.py)
# и строит граф: вершины = именные группы (концепты), ребра = отношения (глаголы, предлоги, притяжание)

import sys
import os

# путь к корню проекта (на один уровень выше папки source/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# путь к папке с переиспользуемыми модулями (Graph, Vertex, Edge, visualize)
TREE_DIR = os.path.join(PROJECT_ROOT, 'rulebased-concept-tree-main')
# добавляем в sys.path, чтобы можно было импортировать graph.higher_dim_graph и т.д.
if TREE_DIR not in sys.path:
    sys.path.insert(0, TREE_DIR)

# deque нужна для BFS при поиске сочинённых элементов (conj)
from collections import deque
# Graph — класс семантического графа (вершины + ребра + эмбеддинги)
from graph.higher_dim_graph import Graph
# visualize_graph — статическая визуализация matplotlib, visualize_graph_interactive — интерактивный HTML (vis.js)
from graph.graph import visualize_graph, visualize_graph_interactive


def is_noun_mk(token):
    # проверяем, что токен -- существительное (NOUN), имя собственное (PROPN) или местоимение (PRON)
    # UPOS-тег хранится в token.pos (универсальная система тегов Universal Dependencies)
    return token.pos in ('NOUN', 'PROPN', 'PRON')


def is_verb_mk(token):
    # проверяем, что токен -- глагол (VERB или AUX)
    # AUX нам не нужен как самостоятельный предикат, поэтому проверяем только VERB
    return token.pos == 'VERB'


def is_prep_mk(token):
    # проверяем, что токен -- предлог (ADP = adposition в UPOS)
    # в македонском: на, во, со, за, од, низ, кон и т.д.
    return token.pos == 'ADP'


def is_conj_mk(token):
    # проверяем, что токен -- союз: сочинительный (CCONJ: и, но, или) или подчинительный (SCONJ: дека, што, кога)
    return token.pos in ('CCONJ', 'SCONJ')


def build_noun_concept_mk(token_id, tokens_map):
    """
    Собирает именную группу (concept) из токена-головы и его модификаторов.
    Голова -- существительное, модификаторы -- прилагательные (amod), детерминаторы (det), числительные (nummod).
    Возвращает строку из лемм, отсортированных по id токена.

    Используем леммы, а не словоформы, потому что в македонском постпозитивный артикль
    меняет форму слова (книгата -> книга), а CLASSLA правильно лемматизирует.
    """
    # берём токен-голову из словаря
    head = tokens_map[token_id]
    # начинаем с самой головы — она всегда входит в концепт
    concept_tokens = [head]

    # перебираем дочерние токены головы
    for child_id in getattr(head, 'children', []):
        # берём дочерний токен из словаря
        child = tokens_map[child_id]
        # приводим deprel к нижнему регистру для сравнения
        deprel = (child.deprel or '').lower()

        # amod / att — прилагательное-модификатор (голем, црвен, интересен)
        # spaCy mk_core_news_lg выдаёт 'att' вместо стандартного UD 'amod'
        if deprel in ('amod', 'att'):
            concept_tokens.append(child)

        # det — детерминатор (овој, тој, секој)
        elif deprel == 'det':
            concept_tokens.append(child)

        # nummod — числительное-модификатор (два, три, пет)
        elif deprel == 'nummod':
            concept_tokens.append(child)

    # сортируем токены по их id, чтобы слова шли в правильном порядке
    # id имеет формат "sent_idx_token_idx", сортировка по строке работает корректно
    concept_tokens.sort(key=lambda t: str(t.id))

    # собираем леммы через пробел — получаем метку концепта
    return ' '.join(t.lemma for t in concept_tokens)


def get_conjuncts_mk(token_id, tokens_map):
    """
    BFS-обход по deprel 'conj' — собирает все сочинённые элементы.
    Если в предложении "Марко и Петар купија", у "Марко" будет дочерний "Петар" с deprel='conj'.
    BFS найдёт обоих.

    Упрощённая версия по сравнению с русской (без эвристик по падежам и xpos).
    """
    # множество id уже обработанных токенов (чтобы не зациклиться)
    result = set()
    # очередь для BFS
    queue = deque([token_id])

    while queue:
        # берём следующий id из очереди
        current_id = queue.popleft()
        # если уже обработан, пропускаем
        if current_id in result:
            continue
        # добавляем в результат
        result.add(current_id)
        # берём токен из словаря
        tok = tokens_map[current_id]

        # перебираем дочерние токены
        for child_id in getattr(tok, 'children', []):
            child = tokens_map[child_id]
            child_deprel = (child.deprel or '').lower()
            # если дочерний элемент привязан через deprel 'conj', добавляем в очередь
            if child_deprel == 'conj':
                queue.append(child_id)

        # проверяем, не привязан ли сам токен к родителю через conj
        head_id = getattr(tok, 'head', None)
        this_deprel = (tok.deprel or '').lower()
        # если текущий токен сам является конъюнктом (deprel='conj'), добавляем его родителя
        if this_deprel == 'conj' and head_id in tokens_map:
            queue.append(head_id)

    return result


def find_cc_label_mk(token_id, tokens_map):
    """
    Ищет союз (deprel='cc') среди дочерних узлов токена.
    Например, для "Марко и Петар" у токена "Петар" будет дочерний "и" с deprel='cc'.
    Возвращает лемму союза или None, если не найден.
    """
    # берём токен из словаря
    tok = tokens_map[token_id]

    # перебираем дочерние узлы
    for child_id in getattr(tok, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'cc' — координирующий союз
        if child_deprel == 'cc':
            # возвращаем лемму союза (и, но, или)
            return child.lemma

    # союз не найден
    return None


def _find_subject(verb_id, tokens_map):
    """
    Ищет подлежащее глагола через deprel 'nsubj' среди его дочерних узлов.
    Возвращает список id токенов-подлежащих.
    """
    # берём токен глагола из словаря
    verb = tokens_map[verb_id]
    # сюда собираем найденные подлежащие
    subjects = []

    # перебираем дочерние узлы глагола
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'nsubj' — именное подлежащее
        if child_deprel == 'nsubj':
            subjects.append(child_id)

    return subjects


def _find_object(verb_id, tokens_map):
    """
    Ищет прямое дополнение глагола через deprel 'obj' / 'dobj' / 'pobj' среди его дочерних узлов.
    spaCy mk_core_news_lg выдаёт 'dobj' и 'pobj' вместо стандартного UD 'obj'.
    Обрабатывает дублирование клитик: если у глагола есть и клитика (PRON с deprel='obj'),
    и полное существительное (NOUN с deprel='obj'), оставляем только существительное.
    """
    # берём токен глагола из словаря
    verb = tokens_map[verb_id]
    # сюда собираем все токены с подходящим deprel
    raw_objects = []

    # перебираем дочерние узлы глагола
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'obj' / 'dobj' / 'pobj' — прямое дополнение
        # spaCy использует 'dobj' (direct object) и 'pobj' (prepositional object)
        if child_deprel in ('obj', 'dobj', 'pobj'):
            raw_objects.append(child_id)

    # обрабатываем клитики: го, ја, ги — местоимения-дубли полного существительного
    # если есть и клитика (PRON), и полное существительное (NOUN/PROPN) с одинаковым deprel='obj',
    # оставляем только существительное (оно несёт больше смысловой информации)
    nouns = [oid for oid in raw_objects if tokens_map[oid].pos in ('NOUN', 'PROPN')]
    pronouns = [oid for oid in raw_objects if tokens_map[oid].pos == 'PRON']

    # если есть хотя бы одно полное существительное, убираем клитики
    if nouns and pronouns:
        return nouns

    # если только клитики или только существительные — возвращаем всё
    return raw_objects


def _find_oblique(verb_id, tokens_map):
    """
    Ищет косвенные дополнения глагола через deprel 'iobj' / 'obl' / 'pozv'.
    spaCy mk_core_news_lg выдаёт нестандартные deprel: 'pozv' (вместо 'obl').
    В македонском 'obl' часто используется для предложных групп (на + N, со + N).
    """
    # берём токен глагола из словаря
    verb = tokens_map[verb_id]
    # сюда собираем косвенные дополнения
    obliques = []

    # перебираем дочерние узлы глагола
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'iobj' — косвенное дополнение (дательный падеж)
        # deprel 'obl' — обстоятельство или предложное дополнение (стандарт UD)
        # deprel 'pozv' — нестандартная метка spaCy для обстоятельства места
        if child_deprel in ('iobj', 'obl', 'pozv'):
            obliques.append(child_id)

    return obliques


def _find_preposition(token_id, tokens_map):
    """
    Ищет предлог (deprel='case') среди дочерних узлов существительного.
    В Universal Dependencies предлоги — дочерние узлы существительного, а не наоборот.
    Пример: "на пазар" — "на" является дочерним узлом "пазар" с deprel='case'.
    """
    # берём токен существительного из словаря
    tok = tokens_map[token_id]

    # перебираем дочерние узлы
    for child_id in getattr(tok, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'case' — маркер падежа (предлог)
        if child_deprel == 'case':
            # возвращаем лемму предлога
            return child.lemma

    # предлог не найден
    return None


def _check_negation(verb_id, tokens_map):
    """
    Проверяет, есть ли отрицание "не" у глагола.
    Частица "не" обычно имеет deprel='advmod' и лемму "не".
    Возвращает True, если глагол отрицается.
    """
    # берём токен глагола из словаря
    verb = tokens_map[verb_id]

    # перебираем дочерние узлы глагола
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # проверяем: лемма "не" и deprel 'advmod' (стандарт UD) или 'neg' (spaCy)
        if child.lemma == 'не' and child_deprel in ('advmod', 'neg'):
            return True

    # отрицания нет
    return False


def _find_xcomp_verb(verb_id, tokens_map):
    """
    Ищет дочерний глагол с deprel='xcomp' для составных конструкций.
    В македонском: "почна да готви" — "готви" будет xcomp у "почна".
    Возвращает id дочернего глагола или None.
    """
    # берём токен глагола из словаря
    verb = tokens_map[verb_id]

    # перебираем дочерние узлы
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # deprel 'xcomp' — открытое клаузальное дополнение (составной глагол)
        if child_deprel == 'xcomp' and is_verb_mk(child):
            return child_id

    # составного глагола нет
    return None


def _find_prep_groups(verb_id, tokens_map):
    """
    Ищет предложные группы в структуре spaCy: verb -> prep -> noun(pobj/iobj).
    spaCy mk_core_news_lg строит дерево иначе, чем Universal Dependencies:
    предлог — дочерний узел глагола с deprel='prep',
    а существительное — дочерний узел предлога с deprel='pobj' или 'iobj'.

    Возвращает список кортежей (prep_lemma, noun_id).
    """
    verb = tokens_map[verb_id]
    prep_groups = []

    # ищем дочерние узлы глагола с deprel='prep'
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()

        # предлог в spaCy-схеме: deprel='prep' и UPOS=ADP
        if child_deprel == 'prep' and is_prep_mk(child):
            prep_lemma = child.lemma
            # ищем дочерние существительные предлога
            for grandchild_id in getattr(child, 'children', []):
                grandchild = tokens_map[grandchild_id]
                gc_deprel = (grandchild.deprel or '').lower()
                # pobj / iobj — существительное, управляемое предлогом
                if gc_deprel in ('pobj', 'iobj', 'obj', 'dobj') and is_noun_mk(grandchild):
                    prep_groups.append((prep_lemma, grandchild_id))

    return prep_groups


def _find_compound_verb_spacy(verb_id, tokens_map):
    """
    Ищет составной глагол в структуре spaCy: verb -> aux("да") -> verb(relcl).
    spaCy mk_core_news_lg кодирует "почна да готви" как:
    "сакаше" -> "да"(aux) -> "готви"(relcl).

    Возвращает id дочернего глагола или None.
    """
    verb = tokens_map[verb_id]

    # ищем дочерний "да" с deprel='aux'
    for child_id in getattr(verb, 'children', []):
        child = tokens_map[child_id]
        child_deprel = (child.deprel or '').lower()
        # "да" как вспомогательная частица
        if child_deprel == 'aux' and child.lemma == 'да':
            # ищем глагол среди дочерних "да"
            for grandchild_id in getattr(child, 'children', []):
                grandchild = tokens_map[grandchild_id]
                if is_verb_mk(grandchild):
                    return grandchild_id

    return None


def extract_predicates_mk(tokens_map):
    """
    Извлекает предикатные структуры из предложения.
    Для каждого глагола находит:
    - подлежащие (nsubj)
    - прямые дополнения (obj/dobj/pobj), с фильтрацией клитик
    - косвенные дополнения (iobj, obl, pozv)
    - предложные группы (prep -> pobj в spaCy-схеме)
    - отрицание (не)
    - составные глаголы (xcomp или da+verb в spaCy-схеме)

    Возвращает список словарей с ключами:
    verb_id, verb_label, subjects, objects, obliques, prep_groups.
    """
    # сюда собираем все найденные предикаты
    predicates = []

    # собираем id глаголов, которые уже обработаны как часть составного глагола
    # чтобы не создавать для них отдельный предикат
    skip_verbs = set()

    # первый проход: находим составные глаголы и помечаем дочерние
    for tid, tok in tokens_map.items():
        if not is_verb_mk(tok):
            continue
        # проверяем xcomp (стандарт UD)
        xcomp_id = _find_xcomp_verb(tid, tokens_map)
        if xcomp_id is not None:
            skip_verbs.add(xcomp_id)
        # проверяем da+verb (spaCy-схема)
        compound_id = _find_compound_verb_spacy(tid, tokens_map)
        if compound_id is not None:
            skip_verbs.add(compound_id)

    # второй проход: строим предикаты
    for tid, tok in tokens_map.items():
        # пропускаем всё, что не глагол
        if not is_verb_mk(tok):
            continue

        # пропускаем глаголы, которые являются частью составного (xcomp или da+verb)
        tok_deprel = (tok.deprel or '').lower()
        if tid in skip_verbs:
            continue
        if tok_deprel == 'xcomp':
            continue

        # формируем метку глагола из леммы
        verb_label = tok.lemma

        # проверяем, есть ли составной глагол (xcomp — стандарт UD)
        xcomp_id = _find_xcomp_verb(tid, tokens_map)
        # проверяем da+verb (spaCy-схема)
        compound_id = _find_compound_verb_spacy(tid, tokens_map)
        # берём тот, который нашёлся
        secondary_verb_id = xcomp_id or compound_id

        if secondary_verb_id is not None:
            # объединяем леммы: "почна да готви" -> "сака готви"
            secondary_tok = tokens_map[secondary_verb_id]
            verb_label = verb_label + ' ' + secondary_tok.lemma

        # проверяем отрицание
        if _check_negation(tid, tokens_map):
            # добавляем "не " перед меткой глагола
            verb_label = 'не ' + verb_label

        # ищем аргументы глагола
        subjects = _find_subject(tid, tokens_map)
        objects = _find_object(tid, tokens_map)
        obliques = _find_oblique(tid, tokens_map)

        # если у глагола нет подлежащего и он зависит от другого глагола
        # (deprel='conj' или 'relcl'), наследуем подлежащее родителя
        # пример: "Детето трчаше ... и се смееше" — "смееше" наследует "Детето"
        if not subjects:
            parent_id = tok.head
            if parent_id in tokens_map and is_verb_mk(tokens_map[parent_id]):
                subjects = _find_subject(parent_id, tokens_map)

        # ищем предложные группы (spaCy-схема: verb -> prep -> noun)
        prep_groups = _find_prep_groups(tid, tokens_map)

        # если есть составной глагол, добавляем его аргументы тоже
        if secondary_verb_id is not None:
            # подлежащие обычно общие, но дополнения могут быть свои
            objects.extend(_find_object(secondary_verb_id, tokens_map))
            obliques.extend(_find_oblique(secondary_verb_id, tokens_map))
            prep_groups.extend(_find_prep_groups(secondary_verb_id, tokens_map))

        # собираем предикат в словарь
        predicates.append({
            'verb_id': tid,
            'verb_label': verb_label,
            'subjects': subjects,
            'objects': objects,
            'obliques': obliques,
            'prep_groups': prep_groups,
        })

    return predicates


def build_mk_graph(sentences):
    """
    Главная функция: принимает список объектов Sentence, строит и возвращает объект Graph.

    Алгоритм:
    1. Для каждого предложения: перенумеровываем id токенов (sent_idx_token_id) для уникальности
    2. Строим children для каждого токена (Token.children не существует по умолчанию)
    3. Извлекаем предикаты (глагол + аргументы)
    4. Для каждого предиката: создаём вершины (именные группы) и ребра (отношения)
    5. Отдельно обрабатываем possessive-конструкции ("на + N" = притяжание)
    6. Дедуплицируем ребра
    """
    # промежуточное хранилище: token_id -> {"label": строка-концепт, "type": "noun"}
    vertices_info = {}
    # промежуточное хранилище: список кортежей (src_token_id, tgt_token_id, edge_label)
    edges_info = []

    # перебираем все предложения
    for sent_idx, sent in enumerate(sentences):
        # берём список токенов предложения
        tokens = sent.tokens

        # перенумеровываем id токенов: добавляем префикс "sent_idx_"
        # без этого id из разных предложений будут конфликтовать (1, 2, 3 в каждом)
        for tok in tokens:
            # запоминаем оригинальный id для пересчёта head
            original_id = tok.id
            # новый уникальный id: "0_1", "0_2", "1_1", "1_2" и т.д.
            tok.id = f"{sent_idx}_{original_id}"

        # перенумеровываем head: тоже добавляем префикс
        for tok in tokens:
            # head хранит оригинальный (числовой) id родителя
            head_val = tok.head
            if head_val == 0 or head_val == '0':
                # head=0 означает корень предложения, оставляем как есть
                tok.head = 0
            else:
                # пересчитываем head с префиксом
                tok.head = f"{sent_idx}_{head_val}"

        # строим словарь token_id -> Token для быстрого поиска
        tokens_map = {tok.id: tok for tok in tokens}

        # инициализируем children для каждого токена (пустой список)
        # Token.children не существует в sent_class.py, нужно добавить вручную
        for tok in tokens:
            tok.children = []

        # заполняем children: для каждого токена добавляем его id в children его head-а
        for tok in tokens:
            head_id = tok.head
            # если head есть в нашем словаре (не корень), добавляем текущий токен в children родителя
            if head_id in tokens_map:
                tokens_map[head_id].children.append(tok.id)

        # извлекаем предикатные структуры (глаголы + их аргументы)
        predicates = extract_predicates_mk(tokens_map)

        # обрабатываем каждый предикат: создаём вершины и ребра
        for pred in predicates:
            # метка ребра — лемма глагола (возможно с "не " и/или xcomp)
            verb_label = pred['verb_label']

            # регистрируем подлежащие как вершины
            for sid in pred['subjects']:
                if sid not in vertices_info:
                    vertices_info[sid] = {
                        'label': build_noun_concept_mk(sid, tokens_map),
                        'type': 'noun',
                    }

            # регистрируем прямые дополнения как вершины
            for oid in pred['objects']:
                if oid not in vertices_info:
                    vertices_info[oid] = {
                        'label': build_noun_concept_mk(oid, tokens_map),
                        'type': 'noun',
                    }

            # регистрируем косвенные дополнения как вершины
            for oid in pred['obliques']:
                if oid not in vertices_info:
                    vertices_info[oid] = {
                        'label': build_noun_concept_mk(oid, tokens_map),
                        'type': 'noun',
                    }

            # создаём ребра: подлежащее -> прямое дополнение (через глагол)
            for sid in pred['subjects']:
                for oid in pred['objects']:
                    # пропускаем петли (подлежащее = дополнение)
                    if sid == oid:
                        continue
                    edges_info.append((sid, oid, verb_label))

            # создаём ребра: подлежащее -> косвенное дополнение (через глагол + предлог)
            for sid in pred['subjects']:
                for oid in pred['obliques']:
                    # пропускаем петли
                    if sid == oid:
                        continue
                    # ищем предлог у косвенного дополнения (UD-схема: case как дочерний)
                    prep = _find_preposition(oid, tokens_map)
                    if prep:
                        # ребро с предлогом: "отиде на" вместо просто "отиде"
                        edge_label = verb_label + ' ' + prep
                    else:
                        # без предлога — просто глагол
                        edge_label = verb_label
                    edges_info.append((sid, oid, edge_label))

            # обрабатываем предложные группы (spaCy-схема: verb -> prep -> noun)
            # это основной путь для spaCy mk_core_news_lg, где предлог — дочерний узел глагола
            for prep_lemma, noun_id in pred.get('prep_groups', []):
                # регистрируем существительное как вершину
                if noun_id not in vertices_info:
                    vertices_info[noun_id] = {
                        'label': build_noun_concept_mk(noun_id, tokens_map),
                        'type': 'noun',
                    }
                # создаём ребра: подлежащее -> noun (через глагол + предлог)
                for sid in pred['subjects']:
                    if sid == noun_id:
                        continue
                    edge_label = verb_label + ' ' + prep_lemma
                    edges_info.append((sid, noun_id, edge_label))

                # если нет подлежащих, связываем прямые дополнения с предложными
                if not pred['subjects']:
                    for oid in pred['objects']:
                        if oid == noun_id:
                            continue
                        edge_label = verb_label + ' ' + prep_lemma
                        edges_info.append((oid, noun_id, edge_label))

            # если нет подлежащих, но есть дополнения — делаем ребра между прямыми и косвенными
            if not pred['subjects'] and pred['objects'] and pred['obliques']:
                for oid in pred['objects']:
                    for oblid in pred['obliques']:
                        if oid == oblid:
                            continue
                        prep = _find_preposition(oblid, tokens_map)
                        if prep:
                            edge_label = verb_label + ' ' + prep
                        else:
                            edge_label = verb_label
                        edges_info.append((oid, oblid, edge_label))

        # обрабатываем сочинённые конструкции (Марко и Петар -> оба связаны с глаголом)
        # находим все конъюнкты (conj) для подлежащих и дополнений
        for pred in predicates:
            verb_label = pred['verb_label']

            # расширяем подлежащие: если "Марко" — подлежащее, а "Петар" — conj-у "Марко",
            # то "Петар" тоже подлежащее
            expanded_subjects = set()
            for sid in pred['subjects']:
                conj_set = get_conjuncts_mk(sid, tokens_map)
                for cid in conj_set:
                    # берём только существительные (не союзы, не знаки)
                    if is_noun_mk(tokens_map[cid]):
                        expanded_subjects.add(cid)

            # расширяем прямые дополнения аналогично
            expanded_objects = set()
            for oid in pred['objects']:
                conj_set = get_conjuncts_mk(oid, tokens_map)
                for cid in conj_set:
                    if is_noun_mk(tokens_map[cid]):
                        expanded_objects.add(cid)

            # регистрируем новые вершины (которых ещё нет)
            for nid in expanded_subjects | expanded_objects:
                if nid not in vertices_info:
                    vertices_info[nid] = {
                        'label': build_noun_concept_mk(nid, tokens_map),
                        'type': 'noun',
                    }

            # создаём ребра для расширенных аргументов
            for sid in expanded_subjects:
                for oid in expanded_objects:
                    if sid == oid:
                        continue
                    # пропускаем, если оба уже были в исходных подлежащих/дополнениях (уже обработаны выше)
                    if sid in pred['subjects'] and oid in pred['objects']:
                        continue
                    edges_info.append((sid, oid, verb_label))

        # обрабатываем possessive-конструкции: "на + N" = притяжание
        # два варианта дерева:
        # 1. UD-схема: noun(nmod) -> case("на"), где noun(nmod) зависит от head-noun
        # 2. spaCy-схема: noun -> prep("на") -> noun(pobj)

        # вариант 1: UD-схема (nmod + case)
        for tid, tok in tokens_map.items():
            if not is_noun_mk(tok):
                continue
            tok_deprel = (tok.deprel or '').lower()
            if tok_deprel != 'nmod':
                continue
            head_id = tok.head
            if head_id not in tokens_map:
                continue
            head_tok = tokens_map[head_id]
            if not is_noun_mk(head_tok):
                continue
            prep = _find_preposition(tid, tokens_map)
            if prep == 'на':
                if head_id not in vertices_info:
                    vertices_info[head_id] = {
                        'label': build_noun_concept_mk(head_id, tokens_map),
                        'type': 'noun',
                    }
                if tid not in vertices_info:
                    vertices_info[tid] = {
                        'label': build_noun_concept_mk(tid, tokens_map),
                        'type': 'noun',
                    }
                edges_info.append((head_id, tid, 'possessive'))
            elif prep:
                if head_id not in vertices_info:
                    vertices_info[head_id] = {
                        'label': build_noun_concept_mk(head_id, tokens_map),
                        'type': 'noun',
                    }
                if tid not in vertices_info:
                    vertices_info[tid] = {
                        'label': build_noun_concept_mk(tid, tokens_map),
                        'type': 'noun',
                    }
                edges_info.append((head_id, tid, prep))

        # вариант 2: spaCy-схема (noun -> prep -> noun(pobj))
        # "Книгата на Марко": Книгата -> на(prep) -> Марко(pobj)
        for tid, tok in tokens_map.items():
            if not is_noun_mk(tok):
                continue
            # ищем дочерний предлог с deprel='prep'
            for child_id in getattr(tok, 'children', []):
                child = tokens_map[child_id]
                child_deprel = (child.deprel or '').lower()
                if child_deprel != 'prep' or not is_prep_mk(child):
                    continue
                prep_lemma = child.lemma
                # ищем существительное-дочерний узел предлога
                for gc_id in getattr(child, 'children', []):
                    gc = tokens_map[gc_id]
                    gc_deprel = (gc.deprel or '').lower()
                    if gc_deprel in ('pobj', 'iobj', 'obj') and is_noun_mk(gc):
                        # регистрируем обе вершины
                        if tid not in vertices_info:
                            vertices_info[tid] = {
                                'label': build_noun_concept_mk(tid, tokens_map),
                                'type': 'noun',
                            }
                        if gc_id not in vertices_info:
                            vertices_info[gc_id] = {
                                'label': build_noun_concept_mk(gc_id, tokens_map),
                                'type': 'noun',
                            }
                        # "на" = possessive, остальные — предложные ребра
                        if prep_lemma == 'на':
                            edges_info.append((tid, gc_id, 'possessive'))
                        else:
                            edges_info.append((tid, gc_id, prep_lemma))

    # строим объект Graph из промежуточных данных
    graph = Graph()

    # добавляем вершины (по меткам-концептам, с дедупликацией)
    # одна и та же метка может встретиться у разных token_id (одно и то же слово в разных предложениях)
    # в Graph вершина хранится по метке, поэтому дублей не будет
    for vid, attr in vertices_info.items():
        label = attr['label']
        # проверяем, не добавлена ли уже вершина с такой меткой
        if label not in graph.vertices:
            # add_vertex бросает ValueError, если вершина уже есть — страхуемся проверкой
            graph.add_vertex(label, label.split())

    # множество для дедупликации рёбер: (src_label, tgt_label, meaning)
    seen_edges = set()

    # добавляем рёбра
    for src_id, tgt_id, meaning in edges_info:
        # проверяем, что оба token_id зарегистрированы как вершины
        if src_id not in vertices_info or tgt_id not in vertices_info:
            continue

        # получаем строковые метки вершин
        src_label = vertices_info[src_id]['label']
        tgt_label = vertices_info[tgt_id]['label']

        # пропускаем петли (одна вершина)
        if src_label == tgt_label:
            continue

        # дедупликация: не добавляем одинаковые ребра
        edge_key = (src_label, tgt_label, meaning)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # проверяем, что обе вершины существуют в графе
        if src_label in graph.vertices and tgt_label in graph.vertices:
            # add_edge принимает 5 отдельных аргументов, а не объект Edge
            graph.add_edge(src_label, tgt_label, meaning)

    return graph


if __name__ == '__main__':
    # тестовый блок: проверяем модуль на 5 македонских предложениях
    import time

    # импортируем process_text_mk из prepare_mk.py (лежит в той же папке source/)
    from prepare_mk import process_text_mk

    # 5 тестовых предложений разной сложности
    test_sentences = [
        "Марко отиде на пазар.",
        "Марко и Петар купија јаболка и круши.",
        "Детето трчаше низ полето и се смееше.",
        "Таа не сакаше да готви вечера.",
        "Книгата на Марко беше интересна.",
    ]

    # описания для каждого тестового случая
    descriptions = [
        "простое предложение с предложной группой",
        "сочинение подлежащих и дополнений",
        "предложная группа + сочинение глаголов",
        "отрицание + составной глагол с 'да'",
        "possessive-конструкция с 'на'",
    ]

    # сюда собираем весь вывод для записи в файл
    output_lines = []
    output_lines.append("make_graph_mk.py: тестовый запуск")
    output_lines.append(f"дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")

    # обрабатываем все предложения через prepare_mk и строим один граф
    print()
    print("ТЕСТ make_graph_mk.py: построение семантического графа")
    print()

    # обрабатываем каждое предложение отдельно, чтобы показать промежуточные результаты
    all_sentences = []
    for idx, (text, desc) in enumerate(zip(test_sentences, descriptions)):
        header = f"ПРЕДЛОЖЕНИЕ {idx + 1}: {desc}"
        print(header)
        print(f"  текст: {text}")
        output_lines.append(header)
        output_lines.append(f"  текст: {text}")

        # обрабатываем текст через гибридный pipeline CLASSLA + spaCy
        sents = process_text_mk(text)
        all_sentences.extend(sents)

        # выводим токены для отладки
        for sent in sents:
            for tok in sent.tokens:
                tok_line = f"  {tok.id:<4} {tok.form:<15} {tok.lemma:<15} {tok.pos:<8} head={tok.head:<4} deprel={tok.deprel}"
                print(tok_line)
                output_lines.append(tok_line)

        print()
        output_lines.append("")

    # строим граф из всех предложений
    print("строим граф из всех предложений...")
    output_lines.append("строим граф из всех предложений...")

    # повторно обрабатываем тексты, т.к. build_mk_graph меняет id токенов
    all_sentences_for_graph = []
    for text in test_sentences:
        sents = process_text_mk(text)
        all_sentences_for_graph.extend(sents)

    # строим граф
    graph = build_mk_graph(all_sentences_for_graph)

    # выводим вершины
    print()
    print("ВЕРШИНЫ ГРАФА:")
    output_lines.append("")
    output_lines.append("ВЕРШИНЫ ГРАФА:")
    for concept in sorted(graph.vertices.keys()):
        vertex_line = f"  {concept}"
        print(vertex_line)
        output_lines.append(vertex_line)

    # выводим ребра
    print()
    print("РЕБРА ГРАФА:")
    output_lines.append("")
    output_lines.append("РЕБРА ГРАФА:")
    for edge in graph.edges:
        edge_line = f"  {edge.agent_1} --[{edge.meaning}]--> {edge.agent_2}"
        print(edge_line)
        output_lines.append(edge_line)

    # статистика
    print()
    stats = f"итого: {len(graph.vertices)} вершин, {len(graph.edges)} ребер"
    print(stats)
    output_lines.append("")
    output_lines.append(stats)

    # визуализация в интерактивный HTML
    vis_path = os.path.join(PROJECT_ROOT, 'temp', 'test_graph_mk.html')
    visualize_graph_interactive(graph, output=vis_path)
    vis_msg = f"визуализация сохранена в {vis_path}"
    print(vis_msg)
    output_lines.append(vis_msg)

    # сохраняем текстовый вывод
    output_path = os.path.join(PROJECT_ROOT, 'temp', 'phase_d_step6_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"вывод сохранён в {output_path}")
