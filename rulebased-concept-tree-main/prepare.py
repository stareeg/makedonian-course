import subprocess
import os

TREE_TAGGER_CMD_EN = "/Users/mac/Desktop/vscode/gromov/tree_tagger/cmd/tree-tagger-english"
TREE_TAGGER_CMD_RU = "/Users/mac/Desktop/vscode/gromov/tree_tagger/cmd/tree-tagger-russian"

def tag_with_treetagger(text, lang='en'):
    """
    Размечает текст с помощью TreeTagger и возвращает список троек (token, pos, lemma).
    """
    if lang == 'en':
        TREE_TAGGER_CMD = TREE_TAGGER_CMD_EN
    elif lang == 'ru':
        TREE_TAGGER_CMD = TREE_TAGGER_CMD_RU
    else:
        raise ValueError("Unsupported language for TreeTagger")
    process = subprocess.run(
        [TREE_TAGGER_CMD],
        input=text,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if process.returncode != 0:
        raise RuntimeError(f"TreeTagger error: {process.stderr}")

    tags = []
    for line in process.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 3:
            token, pos, lemma = parts
            tags.append((token, pos, lemma))
    return tags


def sentence_to_conllx(sentence, lang='en'):
    """
    Преобразует предложение в формат CoNLL-X с POS-тегами.
    HEAD = 0, DEPREL = _
    """
    tagged_tokens = tag_with_treetagger(sentence, lang=lang)
    lines = []
    for i, (token, pos, lemma) in enumerate(tagged_tokens, start=1):
        # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
        lines.append(f"{i}\t{token}\t{lemma}\t{pos}\t{pos}\t_\t0\t_\t_\t_")
    return "\n".join(lines) + "\n"

def write_conllx_file(sentence, filename="input.conll", lang='en'):
    """
    Сохраняет предложение в файл в формате CoNLL-X с POS-тегами.
    """
    conll_data = sentence_to_conllx(sentence, lang=lang)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(conll_data)
    return os.path.abspath(filename)

if __name__ == "__main__":
    # Пример по умолчанию для теста
    # test_sentence = "The cat and the dogs eat meat and fish."
    test_sentence = "John likes Mary. She likes him too."
    file_path = write_conllx_file(test_sentence, lang='en')
    print(f"Saved CoNLL-X file at {file_path}")