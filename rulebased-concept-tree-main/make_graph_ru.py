from collections import deque, defaultdict
from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph, visualize_graph_interactive
from graph.edge import Edge
from graph.vertex import Vertex


def parse_conllu(conllu_text):
    """
    Парсер CoNLL-U-формата (гибкий: пропускает комментарии и пустые строки).
    Возвращает список токенов с полями: id(int), form, lemma, upos, xpos, feats, head(int), deprel, deps, misc, children(list)
    """
    tokens = []
    for line in conllu_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split('\t')
        if len(parts) != 10:
            continue
        tid = parts[0]
        try:
            tid_int = int(tid)
        except ValueError:
            tid_int = tid  # иногда может быть 1.1 etc.
        head = parts[6]
        try:
            head_int = int(head)
        except ValueError:
            head_int = head
        token = {
            "id": tid_int,
            "form": parts[1],
            "lemma": parts[2],
            "upos": parts[3],
            "xpos": parts[4],
            "feats": parts[5],
            "head": head_int,
            "deprel": parts[7],
            "deps": parts[8],
            "misc": parts[9],
            "children": []
        }
        tokens.append(token)
    id_to_index = {t["id"]: i for i, t in enumerate(tokens)}
    for t in tokens:
        h = t["head"]
        if h in id_to_index:
            tokens[id_to_index[h]]["children"].append(t["id"])
    return tokens


def is_noun(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    # N* — существительные
    # P* — местоимения (она, он, они...)
    return (
        xpos.upper().startswith("N") or
        xpos.upper().startswith("P") or
        pos.upper().startswith("N") or
        pos.upper().startswith("P")
    )


def is_verb(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    return (xpos.upper().startswith("V") or pos.upper().startswith("V"))


def is_prep(token):
    xpos = getattr(token, "xpos", "") or ""
    pos = getattr(token, "pos", "") or ""
    # Treat only real prepositions (e.g., Sp-*) as prepositions, not SENT
    return xpos.upper().startswith("SP") or pos.upper() in {"ADP", "PR"}


def is_conjunction(token):
    xpos = getattr(token, "xpos", "") or ""
    deprel = getattr(token, "deprel", "") or ""
    return xpos.upper().startswith("C") or ("соч" in deprel) or ("conj" in deprel) or ("сравн-союзн" in deprel)


def build_noun_concept(token_id, tokens):
    """
    Строит минимальную различающую именную группу (concept NP).

    Включает:
    - голову (существительное / местоимение)
    - amod (прилагательные)
    - det (детерминаторы)
    - nummod (числительные)
    - advmod, если он модифицирует amod

    Не включает:
    - предложные группы
    - другие существительные
    - инфинитивы
    - комплементы
    """

    head = tokens[token_id]
    concept_tokens = [head]

    for child_id in getattr(head, "children", []):
        child = tokens[child_id]
        deprel = (getattr(child, "deprel", "") or "").lower()

        # Прилагательные
        if deprel == "amod":
            concept_tokens.append(child)

        # Детерминаторы (всё, некоторые и т.п.)
        elif deprel == "det":
            concept_tokens.append(child)

        # Числительные
        elif deprel == "nummod":
            concept_tokens.append(child)

        # Наречия, модифицирующие прилагательные (самые интересные)
        elif deprel == "advmod":
            head_of_child = tokens.get(getattr(child, "head", None))
            if head_of_child and (getattr(head_of_child, "deprel", "") or "").lower() == "amod":
                concept_tokens.append(child)

    # сортировка по id (строковые id вида 0_5 тоже корректно сортируются)
    concept_tokens.sort(key=lambda x: str(x.id))

    return " ".join(getattr(t, "lemma", "") for t in concept_tokens)


def get_conjuncts(token_id, tokens_map):
    """
    Возвращает множество id: сам токен + все конъюнкты, связанные через 'соч'/'conj'/'соч-союзн' и т.п.
    Алгоритм: BFS, идём по детям и по родителю/соседям, если встречаем deprel с 'соч' или 'conj' или встречаем CC-узел.
    """
    result = set()
    q = deque([token_id])
    while q:
        cur = q.popleft()
        if cur in result:
            continue
        result.add(cur)
        tok = tokens_map[cur]
        # дети, которые явно conj-like
        for cid in getattr(tok, "children", []):
            child = tokens_map[cid]
            child_deprel = getattr(child, "deprel", "") or ""
            if "соч" in child_deprel or "conj" in child_deprel or "сравн-союзн" in child_deprel:
                q.append(cid)
        # если у текущего токена есть родитель, и родитель — конъюнкт-узел/союз, то добавляем его других детей
        head = getattr(tok, "head", None)
        if head in tokens_map:
            head_tok = tokens_map[head]
            head_deprel = (getattr(head_tok, "deprel", None) or "").lower()
            # если родитель — союзной природы (напр., 'и' с deprel 'сочин'), добавляем все его "соседей"
            if "соч" in head_deprel or "conj" in head_deprel or is_conjunction(head_tok):
                for sib_id in getattr(head_tok, "children", []):
                    if sib_id not in result:
                        q.append(sib_id)
            # также, если наш токен имеет deprel, содержащую 'соч', то включаем родителя и его другие дочерние имена
            this_deprel = (getattr(tok, "deprel", None) or "").lower()
            if "соч" in this_deprel or "conj" in this_deprel:
                q.append(head)
                for sib_id in getattr(head_tok, "children", []):
                    if sib_id not in result:
                        q.append(sib_id)
    # --- Heuristic: if two or more nouns share the same head and there is a conjunction among the children of that head, include all such sibling nouns ---
    # Find the head of our token
    tok = tokens_map[token_id]
    head_id = getattr(tok, "head", None)
    if head_id in tokens_map:
        head_tok = tokens_map[head_id]
        # collect all noun children of this head
        noun_siblings = [cid for cid in getattr(head_tok, "children", []) if is_noun(tokens_map[cid])]
        # check if there is a conjunction among children
        has_conj = any(is_conjunction(tokens_map[cid]) for cid in getattr(head_tok, "children", []))
        if len(noun_siblings) > 1 and has_conj:
            result.update(noun_siblings)
    return result


def find_cc_label_for_conj_pair(n1, n2, tokens_map):
    """
    Попытка найти форму союза (например 'и') между двумя конъюнктными существительными.
    Ищём общий родитель-союз или cc в детях одного из них.
    """
    t1 = tokens_map[n1]
    t2 = tokens_map[n2]
    # ищем общий родитель, у которого дети включают и n1 и n2 и он — союзной природы
    p1 = getattr(t1, "head", None)
    p2 = getattr(t2, "head", None)
    if p1 == p2 and p1 in tokens_map:
        parent = tokens_map[p1]
        if is_conjunction(parent):
            return getattr(parent, "form", None)
    # иначе проверяем детей каждого на deprel 'соч' или 'cc'
    for cid in getattr(t1, "children", []):
        c = tokens_map[cid]
        if (getattr(c, "deprel", "") or "").lower().startswith("соч") or (getattr(c, "deprel", "") or "").lower() == "cc" or is_conjunction(c):
            return getattr(c, "form", None)
    for cid in getattr(t2, "children", []):
        c = tokens_map[cid]
        if (getattr(c, "deprel", "") or "").lower().startswith("соч") or (getattr(c, "deprel", "") or "").lower() == "cc" or is_conjunction(c):
            return getattr(c, "form", None)
    # fallback
    return "и"


def get_case_from_xpos(xpos: str):
    """
    Возвращает букву падежа из xpos (Ncfsan и т.п.)
    n – nom
    a – acc
    d – dat
    g – gen
    i – instr
    l – loc
    """
    if not xpos or len(xpos) < 5:
        return None
    return xpos[4].lower()


def head_chain_contains(start_id, target_id, tokens_map, max_depth=20):
    """
    Проверяет, встречается ли target_id в head-цепочке start_id.
    """
    current = start_id
    depth = 0

    while depth < max_depth:
        head = getattr(tokens_map[current], "head", None)
        if head == target_id:
            return True
        if head == 0 or head == "0" or head not in tokens_map:
            return False
        current = head
        depth += 1

    return False




def extract_predicates(tokens_map):
    predicates = []

    verb_ids = [tid for tid, t in tokens_map.items() if is_verb(t)]
    noun_ids = [tid for tid, t in tokens_map.items() if is_noun(t)]

    for vid in verb_ids:
        subjects = set()
        objects = set()
        indirect = set()
        prep_links = []

        # Track nouns that are attached to a preposition (for acc)
        acc_with_prep = set()

        for nid in noun_ids:
            case = get_case_from_xpos(getattr(tokens_map[nid], "xpos", ""))
            # case-driven argument extraction (more robust to bad dependency trees)
            if case == "n":
                subjects.add(nid)
            elif case == "a":
                # Check if this accusative noun is attached to a preposition
                parent_id = getattr(tokens_map[nid], "head", None)
                if parent_id in tokens_map and is_prep(tokens_map[parent_id]):
                    acc_with_prep.add(nid)
                objects.add(nid)
            elif case in {"d", "l"}:
                indirect.add(nid)

        # fallback: if no subjects found, use nominative noun as subject
        if not subjects:
            for nid in noun_ids:
                case = get_case_from_xpos(getattr(tokens_map[nid], "xpos", ""))
                if case == "n":
                    subjects.add(nid)

        # --- обработка предложных групп ---
        for nid in noun_ids:
            noun_tok = tokens_map[nid]

            # Case 1: noun -> parent is preposition
            parent_id = getattr(noun_tok, "head", None)
            if parent_id in tokens_map and is_prep(tokens_map[parent_id]):
                prep = tokens_map[parent_id]
                prep_lemma = getattr(prep, "lemma", None)
                # Добавляем предлог к ребру, если аргумент связан с предлогом
                prep_links.append((prep_lemma, nid))
                continue

            # Case 2: noun -> ищем предлог среди NP-internal потомков (BFS только внутри NP)
            # BFS for NP-internal prepositions (only inside the NP)
            to_visit = [(cid, 0) for cid in getattr(noun_tok, "children", [])]
            visited = set()
            prep_found = None
            while to_visit:
                cid, depth = to_visit.pop(0)
                if cid in visited:
                    continue
                visited.add(cid)
                child = tokens_map[cid]
                if is_prep(child):
                    prep_head = getattr(child, "head", None)
                    if prep_head != nid:
                        continue
                    # check if this NP is related to the verb
                    if head_chain_contains(nid, vid, tokens_map, max_depth=5) or head_chain_contains(vid, nid, tokens_map, max_depth=5):
                        prep_found = getattr(child, "lemma", None)
                        break
                # only continue BFS through children that are part of NP modifiers
                child_deprel = getattr(child, "deprel", "")
                if child_deprel in {"amod", "det", "nummod", "advmod"}:
                    to_visit.extend([(gcid, depth + 1) for gcid in getattr(child, "children", [])])
            if prep_found:
                prep_links.append((prep_found, nid))

        predicates.append({
            "verb": vid,
            "subjects": subjects,
            "objects": objects,
            "indirect": indirect,
            "prep_links": prep_links,
            "acc_with_prep": acc_with_prep
        })

    return predicates


def build_ru_graph(sentences):
    """
    Публичный интерфейс: получает список объектов Sentence и возвращает объект Graph.
    """
    vertices = {}
    edges = []

    for sent_idx, sent in enumerate(sentences):
        tokens = sent.tokens

        local_tokens_map = {}
        for t in tokens:
            original_id = t.id
            new_id = f"{sent_idx}_{original_id}"
            t.id = new_id
            local_tokens_map[new_id] = t

        # rebuild heads with sentence prefix
        for t in tokens:
            head = getattr(t, "head", None)
            if head == 0 or head == "0":
                t.head = 0
            else:
                t.head = f"{sent_idx}_{head}"

        tokens_map = local_tokens_map

        # rebuild children per sentence
        for t in tokens:
            t.children = []

        for t in tokens:
            head = getattr(t, "head", None)
            if head in tokens_map:
                tokens_map[head].children.append(t.id)

        root_ids = [t.id for t in tokens if getattr(t, "head", None) == 0 or getattr(t, "head", None) == "0"]
        if not root_ids:
            root_ids = [t.id for t in tokens if is_verb(t)]

        predicates = extract_predicates(tokens_map)

        for pred in predicates:
            vid = pred["verb"]
            verb_label = getattr(tokens_map[vid], "lemma", None) or getattr(tokens_map[vid], "form", None)
            # --- добавим поддержку отрицания ---
            verb_tok = tokens_map[vid]
            neg_children = [tokens_map[cid] for cid in getattr(verb_tok, "children", [])
                            if ((getattr(tokens_map[cid], "deprel", "") or "").lower() == "neg" or getattr(tokens_map[cid], "lemma", "") == "не")]
            neg_prefix = "не " if neg_children else ""
            verb_label = neg_prefix + verb_label

            for sid in pred["subjects"]:
                if sid not in vertices:
                    vertices[sid] = {
                        "label": build_noun_concept(sid, tokens_map),
                        "type": "noun"
                    }

            for oid in pred["objects"] | pred["indirect"]:
                if oid not in vertices:
                    vertices[oid] = {
                        "label": build_noun_concept(oid, tokens_map),
                        "type": "noun"
                    }

            # collect nouns that are already linked via preposition
            prep_nouns = {nid for _, nid in pred.get("prep_links", [])}
            acc_with_prep = pred.get("acc_with_prep", set())

            # --- 1. Добавляем рёбра для прямых объектов (acc) ---
            for sid in pred["subjects"]:
                # subjects that are governed by a preposition should not act as subjects
                if sid in prep_nouns:
                    continue
                for oid in pred["objects"]:
                    # filter: genitive modifiers (NP-of-NP) should not act as verb arguments
                    tok = tokens_map.get(oid)
                    case = get_case_from_xpos(getattr(tok, "xpos", "")) if tok else None
                    head = getattr(tok, "head", None) if tok else None
                    if case == "g" and head in tokens_map and is_noun(tokens_map[head]):
                        continue
                    # Для acc-объекта: если он висит на предлоге, не добавлять предлог к ребру (т.е. ребро только с глаголом)
                    if oid in acc_with_prep:
                        edges.append((sid, oid, verb_label))
                    else:
                        # Проверяем, есть ли предлог, реально относящийся к этому объекту (но для acc мы не добавляем его, если есть)
                        edges.append((sid, oid, verb_label))

            # --- 2. Добавляем рёбра для косвенных объектов (dat, loc) без предлога, если они не прикреплены через предлог ---
            for sid in pred["subjects"]:
                if sid in prep_nouns:
                    continue
                for oid in pred["indirect"]:
                    if oid in prep_nouns:
                        continue
                    tok = tokens_map.get(oid)
                    case = get_case_from_xpos(getattr(tok, "xpos", "")) if tok else None
                    head = getattr(tok, "head", None) if tok else None
                    if case == "g" and head in tokens_map and is_noun(tokens_map[head]):
                        continue
                    edges.append((sid, oid, verb_label))

            # --- 3. Рёбра для предложных групп ---
            for prep, nid in pred.get("prep_links", []):
                # Only attach preposition to the noun it actually modifies
                if nid not in vertices:
                    vertices[nid] = {
                        "label": build_noun_concept(nid, tokens_map),
                        "type": "noun"
                    }
                # Find the subject(s) that are actually connected to this object via the verb
                for sid in pred["subjects"]:
                    # skip if subject is the same as the object
                    if sid == nid:
                        continue
                    # Для acc-объекта, если он прикреплён через предлог — не добавлять ребро с предлогом (оно уже добавлено выше без предлога)
                    tok_oid = tokens_map.get(nid)
                    obj_case = get_case_from_xpos(getattr(tok_oid, "xpos", "")) if tok_oid else None
                    if obj_case == "a" and nid in acc_with_prep:
                        continue
                    edges.append((sid, nid, verb_label + " " + prep))

        # --- генитивные связи (NP of NP) ---
        for nid, tok in tokens_map.items():
            if not is_noun(tok):
                continue

            case = get_case_from_xpos(getattr(tok, "xpos", ""))
            if case != "g":
                continue

            head = getattr(tok, "head", None)

            # Case 1: standard structure (head noun governs genitive noun)
            if head in tokens_map and is_noun(tokens_map[head]):
                if head not in vertices:
                    vertices[head] = {
                        "label": build_noun_concept(head, tokens_map),
                        "type": "noun"
                    }

                if nid not in vertices:
                    vertices[nid] = {
                        "label": build_noun_concept(nid, tokens_map),
                        "type": "noun"
                    }

                edges.append((head, nid, "possessive"))
                continue

            # Case 2: Malt-style reversal: nominative noun is child of genitive noun
            for cid in getattr(tok, "children", []):
                child = tokens_map[cid]
                if not is_noun(child):
                    continue

                child_case = get_case_from_xpos(getattr(child, "xpos", ""))
                if child_case != "n":
                    continue

                if cid not in vertices:
                    vertices[cid] = {
                        "label": build_noun_concept(cid, tokens_map),
                        "type": "noun"
                    }

                if nid not in vertices:
                    vertices[nid] = {
                        "label": build_noun_concept(nid, tokens_map),
                        "type": "noun"
                    }

                edges.append((cid, nid, "possessive"))

        # --- NP prepositional modifiers (например: "дом с крышей") ---
        for nid, tok in tokens_map.items():
            if not is_noun(tok):
                continue

            head = getattr(tok, "head", None)
            if head in tokens_map and is_noun(tokens_map[head]):
                for cid in getattr(tok, "children", []):
                    child = tokens_map[cid]
                    if is_prep(child):
                        prep_label = getattr(child, "lemma", None) or getattr(child, "form", None)

                        if head not in vertices:
                            vertices[head] = {
                                "label": build_noun_concept(head, tokens_map),
                                "type": "noun"
                            }

                        if nid not in vertices:
                            vertices[nid] = {
                                "label": build_noun_concept(nid, tokens_map),
                                "type": "noun"
                            }

                        edges.append((head, nid, prep_label))

    # Build the custom Graph: keys are concept strings (to merge conjuncts with same text)
    graph = Graph()
    label_to_vertex_key = {}  # label -> canonical label used as vertex key in Graph

    for vid, attr in vertices.items():
        label = attr["label"]
        if label not in label_to_vertex_key:
            graph.add_vertex(label, [label])
            label_to_vertex_key[label] = label

    seen_edges = set()

    for src_id, tgt_id, lbl in edges:
        if src_id not in vertices or tgt_id not in vertices:
            continue

        src_label = vertices[src_id]["label"]
        tgt_label = vertices[tgt_id]["label"]

        key = (src_label, tgt_label, lbl)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        if src_label in label_to_vertex_key and tgt_label in label_to_vertex_key:
            graph.add_edge(src_label, tgt_label, lbl)

    return graph


if __name__ == "__main__":
    from sent_class import parse_conll

    sentence_list = parse_conll("output.conll")
    graph_res = build_ru_graph(sentence_list)
    print("Graph edges:")
    for e in graph_res.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")
    visualize_graph_interactive(graph_res, output="graph.html")