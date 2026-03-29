from sent_class import parse_conll

class PleonasticItDetector:
    def __init__(self):
        self.modal_adj = {
            "necessary", "possible", "important", "clear", "obvious",
            "likely", "unlikely", "certain", "true", "false"
        }

        self.raising_verbs = {
            "seem", "appear", "happen", "follow"
        }

        self.weather_verbs = {
            "rain", "snow", "hail", "storm"
        }

        self.time_adj = {
            "late", "early", "dark", "cold", "hot"
        }

    def is_pleonastic(self, it_token, sent) -> bool:
        # 1. it must be subject
        if it_token.deprel not in {"nsubj", "expl"}:
            return False

        head = sent.get_token(it_token.head)
        if not head:
            return False

        # 2. Extraposition: it is ADJ (that|to)
        if head.lemma == "be":
            return self._is_extraposition(it_token, head, sent)

        # 3. Raising verbs: it seems that...
        if head.lemma in self.raising_verbs:
            return self._has_clausal_complement(head, sent)

        # 4. Weather / time
        if head.lemma in self.weather_verbs:
            return True

        if head.lemma == "be":
            adj = self._get_predicative_adj(head, sent)
            if adj and adj.lemma in self.time_adj:
                return True

        return False

    def _is_extraposition(self, it_token, be_token, sent):
        adj = self._get_predicative_adj(be_token, sent)
        if not adj:
            return False

        if adj.lemma not in self.modal_adj:
            return False

        # check for clausal complement
        return self._has_clausal_complement(be_token, sent)

    def _get_predicative_adj(self, be_token, sent):
        for tok in sent.tokens:
            if tok.head == be_token.id and tok.deprel in {"acomp", "xcomp"}:
                return tok
        return None

    def _has_clausal_complement(self, head, sent):
        for tok in sent.tokens:
            if tok.head == head.id and tok.deprel in {"ccomp", "xcomp"}:
                return True
        return False


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
            # Head of NP: common noun or proper noun or pronoun
            if tok.pos not in {"NN", "NNS", "NP"}:
                continue

            np_tokens = self._collect_np_tokens(tok, sent)

            np = {
                "head_id": tok.id,
                "tokens": sorted(np_tokens),
                "head_lemma": tok.lemma,
                "head_form": tok.form,
                "pos": tok.pos,
                "number": self._get_number(tok),
                "person": self._get_person(tok),
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
        if tok.deprel in {"nsubj", "nsubjpass"}:
            return "SUBJ"
        if tok.deprel in {"dobj", "obj"}:
            return "OBJ"
        if tok.deprel == "iobj":
            return "IOBJ"
        if tok.deprel == "pobj":
            return "POBJ"
        return "OTHER"

    def _get_number(self, tok):
        if tok.pos in {"NNS", "NNPS"}:
            return "PL"
        if tok.pos in {"NN", "NNP", "NP"}:
            return "SG"
        if tok.pos == "PP":
            return self._pronoun_number(tok.lemma)
        return "UNK"

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
    def filter(self, pronoun_token, candidates):
        sg_pron = {"i", "he", "she", "it", "him", "her"}
        pl_pron = {"we", "they", "them"}

        if pronoun_token.lemma in sg_pron:
            pron_number = "SG"
        elif pronoun_token.lemma in pl_pron:
            pron_number = "PL"
        else:
            return candidates  # unknown → do not filter

        return [np for np in candidates if np["number"] == pron_number]

def resolve_anaphora_en(conll_file: str):
    """
    Resolves English anaphora in a CoNLL file using rule-based RAP approach.
    Returns list of sentences with tokens, where pronouns have 'form' replaced
    by resolved antecedent if found.
    """
    sentences = parse_conll(conll_file)

    np_extractor = NPExtractor()
    pleonastic = PleonasticItDetector()
    binding = BindingFilter()
    dm = DiscourseModel()
    scorer = SalienceScorer()
    morph_filter = MorphFilter()

    for sent in sentences:
        nps = np_extractor.extract_nps(sent)
        dm.decay(sent.sent_id)

        pronouns = [t for t in sent if t.pos == "PP"]

        for pron in pronouns:
            if pron.lemma == "it" and pleonastic.is_pleonastic(pron, sent):
                continue  # leave pleonastic it unchanged

            candidates = [
                np for np in dm.get_candidates(sent.sent_id)
                if np["sentence_id"] < sent.sent_id
            ]

            filtered = binding.filter_coargument(pron, candidates, sent)
            filtered = morph_filter.filter(pron, filtered)
            scored = scorer.score(pron, filtered, sent)

            if scored:
                best = scored[0][0]
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

    resolved_sentences = resolve_anaphora_en("output.conll")

    # print sentences as text
    for sent in resolved_sentences:
        print(f"SENT {sent.sent_id}: {' '.join([t.form for t in sent])}")

    # write resolved CoNLL to file
    # generate_conll(resolved_sentences, "resolved_output.conll")