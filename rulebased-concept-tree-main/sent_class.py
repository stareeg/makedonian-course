# токен -- одно слово с морфологическим и синтаксическим разбором в формате CoNLL-U
class Token:
    """
    Lightweight dependency token (CoNLL-U style).
    """
    def __init__(
        self,
        id,
        form,
        lemma,
        pos,
        xpos,
        head,
        deprel,
        # поле feats хранит морфологические признаки (падеж, род, число и т.д.)
        # по умолчанию "_" -- пустое значение в CoNLL-U, обратная совместимость сохраняется
        feats="_"
    ):
        # порядковый номер токена в предложении (1-based, как в CoNLL-U)
        self.id = id
        # словоформа -- как слово выглядит в тексте
        self.form = form
        # лемма -- начальная форма слова (из лемматизатора)
        self.lemma = lemma
        # UPOS -- универсальный POS-тег (NOUN, VERB, ADJ и т.д.)
        self.pos = pos
        # XPOS -- языкоспецифичный POS-тег (зависит от модели)
        self.xpos = xpos
        # id токена-родителя в дереве зависимостей (0 = корень предложения)
        self.head = head
        # тип синтаксической связи с родителем (nsubj, obj, amod и т.д.)
        self.deprel = deprel
        # морфологические признаки в формате CoNLL-U, например "Case=Nom|Gender=Masc|Number=Sing"
        # нужны для извлечения падежа (граф) и рода/числа (анафора)
        self.feats = feats

    def __repr__(self):
        return f"Token({self.id}, {self.form}, {self.pos}, head={self.head}, rel={self.deprel})"


# предложение -- набор токенов с общим id и текстом
class Sentence:
    """
    Sentence container used across the RAP pipeline.
    """
    def __init__(self, tokens, sent_id, text=None):
        # список объектов Token в порядке их появления в предложении
        self.tokens = tokens
        # порядковый номер предложения (начинается с 0)
        self.sent_id = sent_id
        # исходный текст предложения (берётся из комментария # text = ... в CoNLL-U)
        self.text = text
        # словарь для быстрого поиска токена по его id
        self._token_index = {t.id: t for t in tokens}

    # возвращает токен по его id или None, если такого нет
    def get_token(self, token_id):
        return self._token_index.get(token_id)

    # позволяет перебирать токены в цикле for tok in sentence
    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return f"Sentence(id={self.sent_id}, tokens={len(self.tokens)}) {self.text}"


# читает CoNLL-U файл и возвращает список объектов Sentence
def parse_conll(path):
    """
    Parse a CoNLL-U file and return a list of Sentence objects.

    Expected format (tab-separated, 10 columns):
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    """

    sentences = []
    tokens = []
    sent_id = 0
    current_text = None

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()

            # строка-комментарий с исходным текстом предложения
            if line.startswith("# text ="):
                current_text = line.replace("# text =", "").strip()
                continue

            # пропускаем любые другие комментарии (# sent_id = ... и т.д.)
            if line.startswith("#"):
                continue

            # пустая строка -- граница между предложениями
            if not line:
                if tokens:
                    sentences.append(
                        Sentence(tokens=tokens, sent_id=sent_id, text=current_text)
                    )
                    sent_id += 1
                    tokens = []
                    current_text = None
                continue

            cols = line.split("\t")
            # нужно минимум 8 столбцов (ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL)
            if len(cols) < 8:
                continue

            # пропускаем составные токены вида "1-2" (multiword tokens в CoNLL-U)
            if "-" in cols[0] or "." in cols[0]:
                continue

            # читаем FEATS из 6-го столбца (индекс 5), если он есть
            feats = cols[5] if len(cols) > 5 else "_"

            tok = Token(
                id=int(cols[0]),
                form=cols[1],
                lemma=cols[2],
                pos=cols[3],
                xpos=cols[4],
                head=int(cols[6]),
                deprel=cols[7],
                feats=feats
            )
            tokens.append(tok)

    # последнее предложение, если файл не заканчивается пустой строкой
    if tokens:
        sentences.append(
            Sentence(tokens=tokens, sent_id=sent_id, text=current_text)
        )

    return sentences


# записывает список Sentence в файл в формате CoNLL-U
def generate_conll(sentences, output_path):
    """
    Writes a list of Sentence objects to a file in CoNLL-U format (10 columns).
    """
    with open(output_path, "w", encoding="utf8") as f:
        for sent in sentences:
            # записываем комментарий с текстом предложения, если он есть
            if sent.text:
                f.write(f"# text = {sent.text}\n")
            for tok in sent.tokens:
                # берём feats напрямую из атрибута токена (по умолчанию "_")
                feats = tok.feats
                # deps и misc пока не используем, ставим заглушку "_"
                deps = getattr(tok, "deps", "_")
                misc = getattr(tok, "misc", "_")
                # собираем 10 столбцов CoNLL-U через табуляцию
                line = "\t".join([
                    str(tok.id),
                    tok.form,
                    tok.lemma,
                    tok.pos,
                    tok.xpos,
                    feats,
                    str(tok.head),
                    tok.deprel,
                    deps,
                    misc
                ])
                f.write(line + "\n")
            # пустая строка между предложениями (стандарт CoNLL-U)
            f.write("\n")


if __name__ == "__main__":
    sentences = parse_conll("output.conll")

    for sent in sentences:
        print(sent.text)
        for tok in sent:
            print(tok)