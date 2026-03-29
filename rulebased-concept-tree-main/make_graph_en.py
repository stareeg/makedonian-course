from collections import deque

from graph.higher_dim_graph import Graph
from graph.graph import visualize_graph, visualize_graph_interactive
from graph.edge import Edge
from graph.vertex import Vertex


def parse_conllu(conllu_text):
    """
    Parse multi-sentence CoNLL-U into a list of sentences.
    Each sentence is a list of token dicts.
    """
    sentences = []
    current = []

    for line in conllu_text.split("\n"):
        line = line.strip()

        if not line:
            if current:
                sentences.append(current)
                current = []
            continue

        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) != 10:
            continue

        token = {
            "id": int(parts[0]) if parts[0].isdigit() else parts[0],
            "form": parts[1],
            "lemma": parts[2],
            "upos": parts[3],
            "xpos": parts[4],
            "feats": parts[5],
            "head": int(parts[6]) if parts[6].isdigit() else parts[6],
            "deprel": parts[7],
            "deps": parts[8],
            "misc": parts[9],
            "children": []
        }

        current.append(token)

    if current:
        sentences.append(current)

    return sentences


def build_noun_concept(token, tokens_map):
    """
    Build a noun concept string from a token, including modifiers and compounds.
    """
    def collect_modifiers(token_id, collected):
        token_obj = tokens_map[token_id]
        # first collect modifiers recursively (so compounds/amod before head)
        for child_id in token_obj.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] in {"compound", "amod"}:
                collect_modifiers(child_id, collected)
        collected.append(token_obj["form"])

    collected = []
    collect_modifiers(token["id"], collected)
    return " ".join(collected)


def get_conjuncts(token_id, tokens_map):
    """
    Given a token id, returns a set of token ids including the token and all its conjuncts.
    """
    noun_upos_tags = {"NN", "NNS", "NNP", "NNPS", "NP"}
    conjuncts = set()
    queue = deque([token_id])
    while queue:
        current = queue.popleft()
        if current in conjuncts:
            continue
        if tokens_map[current]["upos"] in noun_upos_tags:
            conjuncts.add(current)
        else:
            continue
        current_token = tokens_map[current]
        for child_id in current_token.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] == "conj" and child["upos"] in noun_upos_tags:
                queue.append(child_id)
    return conjuncts


def find_cc_label(token_id, tokens_map):
    """
    Find coordinating conjunction label (cc) for the token if any.
    """
    token = tokens_map[token_id]
    for child_id in token.get("children", []):
        child = tokens_map[child_id]
        if child["deprel"] == "cc":
            return child["form"]
    return None


def find_related_nouns(verb_id, tokens_map):
    """
    Find subjects and objects related to the verb_id, traversing dep, prep and conj chains.
    Returns:
        subject_ids (set of token ids)
        object_ids_with_prep (dict mapping obj_id -> prep_chain_str)
    """
    noun_upos_tags = {"NN", "NNS", "NNP", "NNPS", "NP"}
    verb_token = tokens_map[verb_id]

    subject_ids = set()
    for child_id in verb_token.get("children", []):
        child = tokens_map[child_id]
        if child["deprel"] == "nsubj" and child["upos"] in noun_upos_tags:
            subject_ids.update(get_conjuncts(child_id, tokens_map))

    object_ids_with_prep = {}

    def collect_objects(token_id, prep_chain=None):
        if prep_chain is None:
            prep_chain = []
        token = tokens_map[token_id]

        if token["deprel"] in {"obj", "dobj", "pobj", "iobj"} and token["upos"] in noun_upos_tags:
            for obj_id in get_conjuncts(token_id, tokens_map):
                object_ids_with_prep[obj_id] = " ".join(prep_chain)
                # ensure conj descendants also get same prep chain
                for c_id in get_conjuncts(obj_id, tokens_map):
                    object_ids_with_prep[c_id] = " ".join(prep_chain)

        if token["deprel"] == "prep":
            new_prep_chain = prep_chain + [token["form"]]
            for child_id in token.get("children", []):
                collect_objects(child_id, new_prep_chain)

        for child_id in token.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] not in {"prep", "obj", "dobj", "pobj", "conj"}:
                collect_objects(child_id, prep_chain)

    for child_id in verb_token.get("children", []):
        collect_objects(child_id)

    if not subject_ids:
        subject_candidates = set()
        current_id = verb_id
        while True:
            current_token = tokens_map[current_id]
            head_id = current_token["head"]
            if head_id == 0 or head_id not in tokens_map:
                break
            head_token = tokens_map[head_id]
            if head_token["upos"] in noun_upos_tags:
                subject_candidates.add(head_id)
            current_id = head_id

        subtree_tokens = set()
        queue = deque([verb_id])
        while queue:
            cur = queue.popleft()
            subtree_tokens.add(cur)
            for ch in tokens_map[cur].get("children", []):
                queue.append(ch)
        noun_tokens_in_subtree = {tid for tid in subtree_tokens if tokens_map[tid]["upos"] in noun_upos_tags}
        subject_candidates.update(noun_tokens_in_subtree - set(object_ids_with_prep.keys()))

        for subj in subject_candidates:
            subject_ids.update(get_conjuncts(subj, tokens_map))

    subject_ids = subject_ids - set(object_ids_with_prep.keys())

    return subject_ids, object_ids_with_prep


def process_node(token_id, tokens_map, vertices, edges):
    """
    Recursive function to process tokens in a single sentence and build:
      - vertices: dict token_id -> label (string)
      - edges: list of tuples (src_label, tgt_label, label)
    Uses tokens_map local to the sentence.
    """
    token = tokens_map[token_id]

    if token["upos"].startswith("V"):
        subj_ids, obj_ids_with_prep = find_related_nouns(token_id, tokens_map)

        for subj_id in subj_ids:
            if subj_id not in vertices and tokens_map[subj_id]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                subj_token = tokens_map[subj_id]
                vertices[subj_id] = build_noun_concept(subj_token, tokens_map)
            subj_label = vertices.get(subj_id)
            if subj_label is None:
                continue

            for obj_id, prep_chain in obj_ids_with_prep.items():
                if tokens_map[obj_id]["upos"] not in {"NN","NNS","NNP","NNPS","NP"}:
                    continue
                if obj_id not in vertices:
                    obj_token = tokens_map[obj_id]
                    vertices[obj_id] = build_noun_concept(obj_token, tokens_map)
                obj_label = vertices[obj_id]

                verb_label = token["lemma"]
                if prep_chain:
                    verb_label = f"{verb_label} {prep_chain}"
                edges.append((subj_label, obj_label, verb_label))

    if token["upos"].startswith("N"):
        if token_id not in vertices:
            vertices[token_id] = build_noun_concept(token, tokens_map)

        for child_id in token.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] == "poss":
                possessor_group = get_conjuncts(child_id, tokens_map)
                possessed_group = get_conjuncts(token_id, tokens_map)

                for n in possessor_group:
                    if tokens_map[n]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                        if n not in vertices:
                            vertices[n] = build_noun_concept(tokens_map[n], tokens_map)
                for n in possessed_group:
                    if tokens_map[n]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                        if n not in vertices:
                            vertices[n] = build_noun_concept(tokens_map[n], tokens_map)

                for tgt in possessed_group:
                    if tokens_map[tgt]["upos"] not in {"NN","NNS","NNP","NNPS","NP"}:
                        continue
                    for src in possessor_group:
                        if tokens_map[src]["upos"] not in {"NN","NNS","NNP","NNPS","NP"}:
                            continue
                        edges.append((vertices[tgt], vertices[src], "possessive"))

    if token["upos"].startswith("N"):
        conjuncts = get_conjuncts(token_id, tokens_map)
        for cid in conjuncts:
            if tokens_map[cid]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                if cid not in vertices:
                    vertices[cid] = build_noun_concept(tokens_map[cid], tokens_map)
        def get_depth(tid):
            depth = 0
            current = tid
            while True:
                head = tokens_map[current]["head"]
                if head == 0 or head not in tokens_map:
                    break
                current = head
                depth += 1
            return depth

        conjuncts_sorted = sorted(conjuncts, key=get_depth)
        for i in range(len(conjuncts_sorted)):
            for j in range(i + 1, len(conjuncts_sorted)):
                n1 = conjuncts_sorted[i]
                n2 = conjuncts_sorted[j]
                depth1 = get_depth(n1)
                depth2 = get_depth(n2)
                cc_label = find_cc_label(n1, tokens_map) if depth1 <= depth2 else find_cc_label(n2, tokens_map)
                if cc_label is None:
                    cc_label = "conj"
                # add bidirectional conj edges using labels, only if vertices exist
                if n1 in vertices and n2 in vertices:
                    edges.append((vertices[n1], vertices[n2], cc_label))
                    edges.append((vertices[n2], vertices[n1], cc_label))

    if token["upos"].startswith("N"):
        noun_group = get_conjuncts(token_id, tokens_map)
        for child_id in token.get("children", []):
            child = tokens_map[child_id]
            if child["deprel"] == "prep":
                prep_label = child["form"]
                for gc_id in child.get("children", []):
                    gc = tokens_map[gc_id]
                    if gc["deprel"] == "pobj":
                        obj_group = get_conjuncts(gc_id, tokens_map)
                        for n in noun_group:
                            if tokens_map[n]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                                if n not in vertices:
                                    vertices[n] = build_noun_concept(tokens_map[n], tokens_map)
                        for o in obj_group:
                            if tokens_map[o]["upos"] in {"NN","NNS","NNP","NNPS","NP"}:
                                if o not in vertices:
                                    vertices[o] = build_noun_concept(tokens_map[o], tokens_map)
                        for n in noun_group:
                            if tokens_map[n]["upos"] not in {"NN","NNS","NNP","NNPS","NP"}:
                                continue
                            for o in obj_group:
                                if tokens_map[o]["upos"] not in {"NN","NNS","NNP","NNPS","NP"}:
                                    continue
                                edges.append((vertices[n], vertices[o], prep_label))

    for child_id in token.get("children", []):
        process_node(child_id, tokens_map, vertices, edges)


def process_sentence(sent):
    """
    Process a single sentence (Sentence object).
    Returns:
      - vertices: dict token_id -> label (string)
      - edges: list of tuples (src_label, tgt_label, label)
    """
    tokens = sent.tokens
    tokens_map = {t.id: t for t in tokens}
    for t in tokens:
        t.children = []
    for t in tokens:
        head = t.head
        if head in tokens_map:
            tokens_map[head].children.append(t.id)

    tokens_map_dict = {}
    for tid, t in tokens_map.items():
        # Convert Token object to dict with all required fields
        token_dict = {
            "id": t.id,
            "form": t.form,
            "lemma": t.lemma,
            "upos": t.pos,
            "xpos": getattr(t, "xpos", "_"),
            "feats": getattr(t, "feats", "_"),
            "head": t.head,
            "deprel": t.deprel,
            "deps": getattr(t, "deps", "_"),
            "misc": getattr(t, "misc", "_"),
            "children": [c for c in getattr(t, "children", [])]
        }
        tokens_map_dict[tid] = token_dict
    # Build children lists for dicts
    for t in tokens_map_dict.values():
        t["children"] = []
    for t in tokens_map_dict.values():
        head = t["head"]
        if head in tokens_map_dict:
            tokens_map_dict[head]["children"].append(t["id"])

    vertices = {}  # token_id -> label
    edges = []     # (src_label, tgt_label, label)

    root_ids = [t["id"] for t in tokens_map_dict.values() if t["head"] == 0 or t["head"] == "0"]
    for root_id in root_ids:
        process_node(root_id, tokens_map_dict, vertices, edges)

    return vertices, edges


def build_en_graph(sentences):
    """
    Build a Graph from a list of Sentence objects.
    Processes each sentence independently and merges results into a single Graph.
    """
    graph = Graph()

    for sent in sentences:
        vertices, edges = process_sentence(sent)

        for vid, label in vertices.items():
            concept_label = label
            if concept_label not in graph.vertices:
                graph.add_vertex(concept_label, [concept_label])

        for src_label, tgt_label, label in edges:
            if src_label in graph.vertices and tgt_label in graph.vertices:
                try:
                    graph.add_edge(src_label, tgt_label, label)
                except TypeError:
                    graph.add_edge(src_label, tgt_label, label, 1, 0)

    return graph


if __name__ == "__main__":
    from sent_class import parse_conll

    sentence_list = parse_conll("output.conll")
    graph_res = build_en_graph(sentence_list)
    visualize_graph_interactive(graph_res, output="graph.html")