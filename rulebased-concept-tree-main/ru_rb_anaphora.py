from sent_class import parse_conll
import pymorphy3


class NPExtractor:
    """
    Rule-based NP extractor for dependency trees.
    Extracts noun phrases with grammatical roles needed for RAP.
    """

    def extract_nps(self, sent):
        """
        sent: sentence object with attributes:
            - tokens: list of Token
            - get_token(id) -> Token

        returns: list of NP dicts
        """

        nps = []

        for tok in sent.tokens:
            # все существительные MaltParser (теги, начинающиеся на N)
            if not tok.pos.startswith("N"):
                continue

            np_tokens = self._collect_np_tokens(tok, sent)

            np = {
                "head_id": tok.id,
                "tokens": sorted(np_tokens),
                "head_lemma": tok.lemma,
                "head_form": tok.form,
                "pos": tok.pos,
                "number": self._get_number(tok),
                "person": 3,
                "gram_role": self._get_grammatical_role(tok),
                "sentence_id": sent.sent_id
            }

            nps.append(np)

        return nps

    def _collect_np_tokens(self, head, sent):
        """
        Collect determiners, compounds, adjectival modifiers.
        """
        tokens = {head.id}

        for tok in sent.tokens:
            if tok.head == head.id and tok.deprel in {
                "det", "amod", "compound", "nummod", "poss"
            }:
                tokens.add(tok.id)

        return tokens

    def _get_grammatical_role(self, tok):
        # соответствие тегов MaltParser к SUBJ/OBJ
        if tok.deprel == "огранич":
            return "SUBJ"
        if tok.deprel == "предл":
            return "OBJ"
        return "OTHER"

    def _get_number(self, tok):
        if tok.pos.endswith("s") or tok.pos in {"NNS"}:
            return "PL"
        return "SG"

    def _get_person(self, tok):
        if tok.pos == "PRP":
            return self._pronoun_person(tok.lemma)
        return 3

    def _pronoun_number(self, lemma):
        if lemma in {"we", "they"}:
            return "PL"
        if lemma in {"i", "he", "she", "it"}:
            return "SG"
        return "UNK"

    def _pronoun_person(self, lemma):
        if lemma == "i":
            return 1
        if lemma == "you":
            return 2
        return 3

class BindingFilter:
    """
    Implements Binding Theory filters for RAP.
    Currently: Co-argument constraint (Condition B).
    """

    def filter_coargument(self, pronoun_token, candidate_nps, sent):
        """
        Remove NP candidates that are co-arguments of the same predicate
        as the pronoun (Condition B), but only for NPs in the same sentence.
        """
        filtered = []

        pron_head = sent.get_token(pronoun_token.head)
        if not pron_head:
            return candidate_nps

        for np in candidate_nps:
            # NP from previous sentence: keep without filtering
            if np["sentence_id"] != sent.sent_id:
                filtered.append(np)
                continue

            cand_head = sent.get_token(np["head_id"])
            if not cand_head:
                filtered.append(np)  # keep if head not found
                continue

            # same governing predicate in the same sentence
            if cand_head.head == pron_head.id and pronoun_token.head == pron_head.id:
                continue

            filtered.append(np)

        return filtered


class DiscourseModel:
    """
    Minimal discourse model for RAP.
    Keeps track of active NP candidates with sentence-based decay.
    """

    def __init__(self, max_sent_distance=4):
        self.max_sent_distance = max_sent_distance
        self.entities = []  # list of dicts: NP + salience + last_seen

    def add_nps(self, nps, current_sent_id):
        """
        Add new NPs from the current sentence to the discourse model.
        Initial salience is assigned later; here we just register them.
        """
        for np in nps:
            entry = {
                "np": np,
                "salience": 0,
                "last_seen": current_sent_id
            }
            self.entities.append(entry)

    def decay(self, current_sent_id):
        """
        Apply sentence-based decay and remove stale entities.
        """
        new_entities = []

        for ent in self.entities:
            distance = current_sent_id - ent["last_seen"]

            if distance > self.max_sent_distance:
                continue

            ent["salience"] = ent["salience"] / (2 ** distance)
            new_entities.append(ent)

        self.entities = new_entities

    def get_candidates(self, current_sent_id):
        """
        Return NP candidates within discourse window.
        """
        candidates = []

        for ent in self.entities:
            distance = current_sent_id - ent["last_seen"]
            if distance <= self.max_sent_distance:
                candidates.append(ent["np"])

        return candidates


class SalienceScorer:
    """
    Computes salience weights for NP candidates in RAP.
    """

    ROLE_WEIGHTS = {
        "SUBJ": 80,
        "OBJ": 50,
        "IOBJ": 40,
        "POBJ": 40,
        "OTHER": 0
    }

    RECENCY_WEIGHT = 100
    HEAD_WEIGHT = 80
    PARALLELISM_BONUS = 35

    def score(self, pronoun_token, candidates, sent):
        scored = []

        for np in candidates:
            weight = 0

            # sentence recency
            if np["sentence_id"] == sent.sent_id:
                weight += self.RECENCY_WEIGHT

            # grammatical role
            weight += self.ROLE_WEIGHTS.get(np["gram_role"], 0)

            # head NP emphasis
            # simple approximation: if NP not nested (tokens == 1)
            if len(np["tokens"]) == 1:
                weight += self.HEAD_WEIGHT

            # parallelism: if pronoun role matches NP role
            pron_role = self._get_pronoun_role(pronoun_token)
            if pron_role == np["gram_role"]:
                weight += self.PARALLELISM_BONUS

            scored.append((np, weight))

        # sort descending by weight
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_pronoun_role(self, pron_token):
        if pron_token.deprel in {"nsubj", "nsubjpass"}:
            return "SUBJ"
        if pron_token.deprel in {"dobj", "obj"}:
            return "OBJ"
        if pron_token.deprel == "iobj":
            return "IOBJ"
        if pron_token.deprel == "pobj":
            return "POBJ"
        return "OTHER"

class MorphFilter:
    """
    Morphological filter for Russian:
    keeps only NP candidates matching pronoun in gender and number
    """
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def filter(self, pronoun_token, candidates):
        # Определяем род и число для местоимения
        pron_feats = self.morph.parse(pronoun_token.form)[0]
        pron_gender = pron_feats.tag.gender
        pron_number = pron_feats.tag.number

        if pron_gender is None and pron_number is None:
            # если не удалось определить — не фильтруем
            return candidates

        filtered = []
        for np in candidates:
            # Разбираем head NP через pymorphy3
            np_feats = self.morph.parse(np["head_form"])[0]
            np_gender = np_feats.tag.gender
            np_number = np_feats.tag.number

            # Сравниваем по числу и роду
            if pron_number and np_number != pron_number:
                continue
            if pron_gender and np_gender != pron_gender:
                continue

            filtered.append(np)

        return filtered

def resolve_anaphora_ru(conll_file: str):
    """
    Resolves Russian anaphora in a CoNLL file using rule-based RAP approach.
    Returns list of sentences with tokens, where pronouns have 'form' replaced
    by resolved antecedent if found.
    """
    sentences = parse_conll(conll_file)

    np_extractor = NPExtractor()
    binding = BindingFilter()
    dm = DiscourseModel()
    scorer = SalienceScorer()
    morph_filter = MorphFilter()

    for sent in sentences:
        nps = np_extractor.extract_nps(sent)
        dm.decay(sent.sent_id)

        # распознаём местоимения P-* вместо PP
        pronouns = [t for t in sent if t.pos.startswith("P-")]

        for pron in pronouns:
            # берем кандидатов из предыдущих предложений
            candidates = [
                np for np in dm.get_candidates(sent.sent_id)
                if np["sentence_id"] < sent.sent_id
            ]

            # BindingFilter
            filtered = binding.filter_coargument(pron, candidates, sent)

            # MorphFilter
            filtered = morph_filter.filter(pron, filtered)

            # Salience
            scored = scorer.score(pron, filtered, sent)

            if scored:
                best = scored[0][0]

                # морфологический анализ местоимения
                pron_parse = morph_filter.morph.parse(pron.form)[0]

                target_case = pron_parse.tag.case
                target_number = pron_parse.tag.number
                target_gender = pron_parse.tag.gender

                # анализ антецедента
                antecedent_parse = morph_filter.morph.parse(best["head_form"])[0]

                # лемма в нормальной форме (именительный падеж)
                pron.lemma = antecedent_parse.normal_form

                # собираем граммемы для согласования формы
                grammemes = set()

                if target_case:
                    grammemes.add(target_case)
                if target_number:
                    grammemes.add(target_number)
                if target_gender:
                    grammemes.add(target_gender)

                # пытаемся просклонять антецедент
                inflected = antecedent_parse.inflect(grammemes)

                if inflected:
                    pron.form = inflected.word
                else:
                    # fallback — используем исходную форму
                    pron.form = best["head_form"]

                sent.text = " ".join(t.form for t in sent.tokens)

        dm.add_nps(nps, sent.sent_id)

    return sentences


def sentences_to_conll(sentences):
    """
    Convert list of Sentence objects to CoNLL formatted strings.
    Returns list of strings for each sentence.
    """
    conll_lines = []

    for sent in sentences:
        conll_lines.append(f"# text = {sent.text}")
        for tok in sent.tokens:
            # Assuming Token has attributes: id, form, lemma, pos, xpos, feats, head, deprel, deps, misc
            # Using '_' for xpos, feats, deps, misc if not present
            xpos = getattr(tok, 'xpos', '_')
            feats = getattr(tok, 'feats', '_')
            deps = getattr(tok, 'deps', '_')
            misc = getattr(tok, 'misc', '_')
            line = "\t".join([
                str(tok.id),
                tok.form,
                tok.lemma,
                tok.pos,
                xpos,
                feats,
                str(tok.head),
                tok.deprel,
                deps,
                misc
            ])
            conll_lines.append(line)
        conll_lines.append("")  # empty line between sentences

    return conll_lines


if __name__ == "__main__":
    from sent_class import generate_conll

    resolved_sentences = resolve_anaphora_ru("output.conll")

    # print sentences as text
    for sent in resolved_sentences:
        print(f"SENT {sent.sent_id}: {' '.join([t.form for t in sent])}")

    with open("resolved_output.conll", "w") as f:
        f.write("\n".join(sentences_to_conll(resolved_sentences)))