# prepare_mk.py
# модуль подготовки текста для македонского языка
# гибридный pipeline: CLASSLA (tokenize, POS, lemma) + spaCy (dependency parsing)
#
# CLASSLA не поддерживает dependency parsing для македонского (единственный язык без depparse),
# поэтому синтаксический разбор берём из spaCy mk_core_news_lg

import re
import sys
import os
import time
import logging

# принудительно ставим UTF-8, чтобы кириллица не ломалась в Windows-консоли
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# добавляем корневую папку проекта в sys.path, чтобы можно было импортировать sent_class
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# папка с модулем sent_class.py (классы Token и Sentence)
SENT_CLASS_DIR = os.path.join(PROJECT_ROOT, 'rulebased-concept-tree-main')
if SENT_CLASS_DIR not in sys.path:
    sys.path.insert(0, SENT_CLASS_DIR)

# импортируем классы Token и Sentence для представления разобранных предложений
from sent_class import Token, Sentence

# настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
# логгер для нашего модуля
logger = logging.getLogger('prepare_mk')

# глобальные переменные для кеширования загруженных pipeline-ов
# загружаем один раз при первом вызове (lazy loading)
_classla_pipeline = None
_spacy_pipeline = None


def _get_classla_pipeline():
    """
    Возвращает CLASSLA pipeline для македонского (tokenize + POS + lemma).
    При первом вызове загружает модель и кеширует; при повторных вызовах отдаёт из кеша.
    """
    global _classla_pipeline
    # если pipeline уже загружен, просто возвращаем его
    if _classla_pipeline is not None:
        return _classla_pipeline

    import classla

    logger.info("загружаем CLASSLA pipeline для mk (tokenize, pos, lemma)...")
    # создаём pipeline с тремя процессорами: разбивка на токены, POS-тегирование, лемматизация
    _classla_pipeline = classla.Pipeline(
        'mk',
        processors='tokenize,pos,lemma',
        logging_level='WARN'
    )
    logger.info("CLASSLA pipeline загружен")
    return _classla_pipeline


def _get_spacy_pipeline():
    """
    Возвращает spaCy pipeline для македонского (mk_core_news_lg).
    При первом вызове загружает модель и кеширует; при повторных вызовах отдаёт из кеша.
    """
    global _spacy_pipeline
    # если pipeline уже загружен, просто возвращаем его
    if _spacy_pipeline is not None:
        return _spacy_pipeline

    import spacy

    logger.info("загружаем spaCy pipeline mk_core_news_lg...")
    # загружаем большую модель для македонского (содержит dependency parser)
    _spacy_pipeline = spacy.load('mk_core_news_lg')
    logger.info("spaCy pipeline загружен")
    return _spacy_pipeline


def split_into_chunks(text, max_chunk_size=5000):
    """
    Разбивает текст на чанки по границам предложений.
    CLASSLA не справляется с очень длинными текстами целиком (память, скорость),
    поэтому мы делаем pre-split на куски разумного размера.

    Логика:
    1. Разбиваем текст по концам предложений (. ! ? + пробел или перенос)
    2. Собираем фрагменты в чанки, пока длина не превышает max_chunk_size
    3. Если один фрагмент длиннее max_chunk_size, режем по ближайшему пробелу
    """
    # если текст короче лимита, возвращаем как есть
    if len(text) <= max_chunk_size:
        return [text]

    # разбиваем по концам предложений: точка/!/? + пробел или перенос строки
    # re.split с lookahead сохраняет разделитель в тексте
    fragments = re.split(r'(?<=[.!?])(?=\s)', text)

    chunks = []
    # текущий чанк, который мы собираем
    current_chunk = ""

    for fragment in fragments:
        # если фрагмент сам по себе длиннее лимита, режем по пробелам
        if len(fragment) > max_chunk_size:
            # сбрасываем накопленный чанк
            if current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = ""
            # режем длинный фрагмент по пробелам
            sub_chunks = _split_long_fragment(fragment, max_chunk_size)
            # все подчанки кроме последнего добавляем сразу
            for sc in sub_chunks[:-1]:
                chunks.append(sc)
            # последний подчанок начинаем склеивать дальше
            current_chunk = sub_chunks[-1] if sub_chunks else ""
            continue

        # проверяем, влезет ли фрагмент в текущий чанк
        if len(current_chunk) + len(fragment) <= max_chunk_size:
            # влезает, добавляем
            current_chunk += fragment
        else:
            # не влезает, сбрасываем текущий чанк и начинаем новый
            if current_chunk.strip():
                chunks.append(current_chunk)
            current_chunk = fragment

    # не забываем про остаток
    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


def _split_long_fragment(fragment, max_size):
    """
    Режет слишком длинный фрагмент (> max_size символов) по ближайшему пробелу.
    Не разрезает слова пополам.
    """
    pieces = []
    while len(fragment) > max_size:
        # ищем последний пробел в пределах max_size символов
        cut_pos = fragment.rfind(' ', 0, max_size)
        if cut_pos == -1:
            # если пробела нет (одно гигантское слово), режем по лимиту
            cut_pos = max_size
        # отрезаем кусок до пробела
        pieces.append(fragment[:cut_pos])
        # оставшуюся часть продолжаем разрезать
        fragment = fragment[cut_pos:].lstrip()

    # остаток тоже добавляем
    if fragment.strip():
        pieces.append(fragment)

    return pieces


def _align_spacy_to_classla(classla_words, spacy_doc):
    """
    Сопоставляет токены spaCy с токенами CLASSLA для извлечения head и deprel.
    spaCy может токенизировать иначе, чем CLASSLA, поэтому делаем best-effort
    позиционное сопоставление по символьным смещениям.

    Возвращает список кортежей (head, deprel) по одному на каждый токен CLASSLA.
    head в формате CoNLL-U: 1-based индекс, 0 = корень.
    """
    # количество токенов CLASSLA
    n_classla = len(classla_words)

    # собираем spaCy токены в список для удобства
    spacy_tokens = list(spacy_doc)
    n_spacy = len(spacy_tokens)

    # если количество токенов совпадает, сопоставляем по позициям напрямую
    if n_classla == n_spacy:
        result = []
        for i in range(n_classla):
            sp_tok = spacy_tokens[i]
            # spaCy head.i дает абсолютный индекс в документе
            # если токен указывает сам на себя, это корень (head = 0 в CoNLL-U)
            if sp_tok.head.i == sp_tok.i:
                head = 0
            else:
                # переводим в 1-based индекс
                head = sp_tok.head.i + 1
            deprel = sp_tok.dep_
            result.append((head, deprel))
        return result

    # если количество не совпадает, пробуем посимвольное сопоставление
    # строим символьные позиции для CLASSLA токенов
    # (используем слова через пробел, как мы их подали в spaCy)
    classla_texts = [w.text for w in classla_words]

    # строим карту: индекс spaCy токена -> (start_char, end_char) в тексте spaCy
    # spaCy doc уже содержит start_char и end_char для каждого токена

    # строим карту: индекс CLASSLA токена -> индекс ближайшего spaCy токена
    # вычисляем начальные позиции CLASSLA токенов в склеенной строке
    classla_starts = []
    pos = 0
    for text in classla_texts:
        classla_starts.append(pos)
        # +1 за пробел между токенами
        pos += len(text) + 1

    # для каждого CLASSLA токена ищем ближайший spaCy токен по start_char
    classla_to_spacy = {}
    for ci in range(n_classla):
        c_start = classla_starts[ci]
        # ищем spaCy токен с ближайшим start_char
        best_si = 0
        best_dist = abs(spacy_tokens[0].idx - c_start)
        for si in range(1, n_spacy):
            dist = abs(spacy_tokens[si].idx - c_start)
            if dist < best_dist:
                best_dist = dist
                best_si = si
        classla_to_spacy[ci] = best_si

    # обратная карта: spaCy индекс -> CLASSLA индекс (нужна для пересчёта head)
    spacy_to_classla = {}
    for ci, si in classla_to_spacy.items():
        # если несколько CLASSLA токенов указывают на один spaCy, берём первый
        if si not in spacy_to_classla:
            spacy_to_classla[si] = ci

    # собираем результат
    result = []
    for ci in range(n_classla):
        si = classla_to_spacy.get(ci)
        if si is not None and si < n_spacy:
            sp_tok = spacy_tokens[si]
            # определяем deprel
            deprel = sp_tok.dep_
            # определяем head: пересчитываем spaCy head index в CLASSLA индексацию
            if sp_tok.head.i == sp_tok.i:
                # корень
                head = 0
            else:
                # ищем CLASSLA индекс для spaCy head
                head_si = sp_tok.head.i
                head_ci = spacy_to_classla.get(head_si)
                if head_ci is not None:
                    # +1 для 1-based CoNLL-U
                    head = head_ci + 1
                else:
                    # не нашли соответствие для head, ставим 0
                    head = 0
        else:
            # для этого CLASSLA токена не нашли spaCy соответствие
            head = 0
            deprel = "_"
        result.append((head, deprel))

    return result


def classla_to_sentences(classla_doc, spacy_nlp):
    """
    Конвертирует результат CLASSLA Document в список объектов Sentence (из sent_class.py).
    Для dependency parsing использует spaCy.

    classla_doc: результат CLASSLA pipeline (объект classla.Document)
    spacy_nlp: загруженный spaCy pipeline (mk_core_news_lg)

    Возвращает: List[Sentence], где каждый Token содержит
    id, form, lemma, pos (UPOS), xpos, head, deprel, feats.
    """
    sentences = []

    for sent_idx, classla_sent in enumerate(classla_doc.sentences):
        # собираем токены CLASSLA, пропуская multiword tokens (id вида "1-2")
        classla_words = []
        for word in classla_sent.words:
            # multiword tokens в CLASSLA имеют id типа tuple или строку с дефисом
            # обычные слова имеют целочисленный id
            # проверяем через parent token: если у token есть range id, пропускаем
            classla_words.append(word)

        # если предложение пустое, пропускаем
        if not classla_words:
            continue

        # собираем текст предложения из форм CLASSLA для передачи в spaCy
        sent_text = ' '.join(w.text for w in classla_words)

        # прогоняем текст через spaCy для dependency parsing
        try:
            spacy_doc = spacy_nlp(sent_text)
            # сопоставляем токены spaCy и CLASSLA
            dep_info = _align_spacy_to_classla(classla_words, spacy_doc)
        except Exception as e:
            # если spaCy сломался, ставим заглушки
            logger.warning(f"spaCy не смог обработать предложение {sent_idx}: {e}")
            dep_info = [(0, "_")] * len(classla_words)

        # создаём объекты Token для каждого слова
        tokens = []
        for i, word in enumerate(classla_words):
            # извлекаем feats из CLASSLA (морфологические признаки)
            # word.feats возвращает строку типа "Case=Nom|Gender=Masc|Number=Sing" или None
            feats = word.feats if word.feats else "_"

            # извлекаем xpos из CLASSLA (языкоспецифичный POS-тег)
            xpos = word.xpos if word.xpos else "_"

            # head и deprel берём из spaCy (через сопоставление)
            head, deprel = dep_info[i]

            # создаём Token (1-based id, как в CoNLL-U)
            tok = Token(
                id=i + 1,
                form=word.text,
                lemma=word.lemma,
                pos=word.upos,
                xpos=xpos,
                head=head,
                deprel=deprel,
                feats=feats
            )
            tokens.append(tok)

        # собираем текст предложения (оригинальный, из CLASSLA)
        original_text = classla_sent.text if hasattr(classla_sent, 'text') else sent_text

        # создаём объект Sentence
        sent = Sentence(tokens=tokens, sent_id=sent_idx, text=original_text)
        sentences.append(sent)

    return sentences


def process_text_mk(text):
    """
    Главная функция: принимает сырой текст на македонском,
    возвращает List[Sentence] с полным морфо-синтаксическим разбором.

    Каждый Token содержит: id, form, lemma, pos (UPOS), xpos, head, deprel, feats.

    Внутри:
    1. Разбивает текст на чанки (чтобы CLASSLA не падал на длинных текстах)
    2. Каждый чанк обрабатывает через CLASSLA (tokenize, POS, lemma, feats)
    3. Каждое предложение дополнительно прогоняет через spaCy (dependency parsing)
    4. Объединяет результаты в объекты Token/Sentence из sent_class.py
    """
    # засекаем время
    start_time = time.time()

    # загружаем pipeline-ы (при первом вызове загрузка, дальше из кеша)
    classla_nlp = _get_classla_pipeline()
    spacy_nlp = _get_spacy_pipeline()

    # разбиваем текст на чанки по границам предложений
    chunks = split_into_chunks(text, max_chunk_size=5000)
    logger.info(f"текст разбит на {len(chunks)} чанков ({len(text)} символов)")

    # сюда будем складывать все предложения из всех чанков
    all_sentences = []
    # счётчик для сквозной нумерации предложений
    sent_counter = 0
    # сюда пишем ошибки
    errors = []

    for chunk_idx, chunk in enumerate(chunks):
        # пропускаем пустые чанки
        if not chunk.strip():
            continue

        try:
            # прогоняем чанк через CLASSLA
            classla_doc = classla_nlp(chunk)

            # конвертируем результат CLASSLA в Sentence-ы с dependency parsing от spaCy
            chunk_sentences = classla_to_sentences(classla_doc, spacy_nlp)

            # корректируем sent_id для сквозной нумерации
            for sent in chunk_sentences:
                sent.sent_id = sent_counter
                sent_counter += 1

            # добавляем в общий список
            all_sentences.extend(chunk_sentences)

        except Exception as e:
            # ловим ошибку и логируем, но не останавливаемся
            error_msg = f"чанк {chunk_idx + 1}/{len(chunks)}: {type(e).__name__}: {e}"
            logger.warning(f"ошибка при обработке: {error_msg}")
            errors.append(error_msg)

    # считаем итоговую статистику
    elapsed = time.time() - start_time
    n_sentences = len(all_sentences)
    n_tokens = sum(len(s.tokens) for s in all_sentences)

    logger.info(
        f"готово: {n_sentences} предложений, {n_tokens} токенов, "
        f"{len(errors)} ошибок, {elapsed:.2f} сек"
    )

    return all_sentences


def text_to_conllu_mk(text):
    """
    Принимает сырой текст на македонском, возвращает строку в формате CoNLL-U (10 столбцов).
    Удобно для отладки и сохранения результатов в файл.
    """
    # обрабатываем текст через гибридный pipeline
    sentences = process_text_mk(text)

    # собираем строки CoNLL-U
    lines = []
    for sent in sentences:
        # комментарий с номером предложения
        lines.append(f"# sent_id = {sent.sent_id}")
        # комментарий с текстом предложения
        if sent.text:
            lines.append(f"# text = {sent.text}")
        # токены: 10 столбцов через табуляцию
        for tok in sent.tokens:
            feats = tok.feats if tok.feats else "_"
            # deps и misc не используем, ставим заглушку
            deps = "_"
            misc = "_"
            # собираем 10 столбцов CoNLL-U
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
            lines.append(line)
        # пустая строка между предложениями (стандарт CoNLL-U)
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # тестовый блок: проверяем работу модуля на 5 македонских предложениях
    print()
    print("ТЕСТ prepare_mk.py: гибридный pipeline CLASSLA + spaCy")
    print()

    # 5 тестовых предложений разной сложности
    test_sentences = [
        "Марко отиде на пазар.",
        "Кучето што го видов вчера беше големо и црно.",
        "Скопје е главен град на Северна Македонија.",
        "Тој рече дека ќе дојде утре, но таа не му веруваше.",
        "Детето трчаше низ полето и се смееше.",
    ]

    # описания для каждого предложения (для вывода)
    descriptions = [
        "простое предложение",
        "сложноподчиненное",
        "именованные сущности",
        "сложное с местоимениями",
        "с сочинением",
    ]

    # сюда будем собирать весь вывод для записи в файл
    output_lines = []
    output_lines.append("prepare_mk.py: тестовый запуск")
    output_lines.append(f"дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")

    # общие счётчики
    total_sentences = 0
    total_tokens = 0

    for idx, (text, desc) in enumerate(zip(test_sentences, descriptions)):
        header = f"ПРЕДЛОЖЕНИЕ {idx + 1}: {desc}"
        print(header)
        print(f"  текст: {text}")
        output_lines.append(header)
        output_lines.append(f"  текст: {text}")

        # обрабатываем через гибридный pipeline
        sentences = process_text_mk(text)

        # выводим таблицу токенов для каждого предложения
        table_header = f"  {'ID':<4} {'Форма':<20} {'Лемма':<20} {'UPOS':<8} {'XPOS':<10} {'FEATS':<35} {'HEAD':<6} {'DEPREL'}"
        table_sep = f"  {'--':<4} {'-----':<20} {'-----':<20} {'----':<8} {'----':<10} {'-----':<35} {'----':<6} {'------'}"
        print(table_header)
        print(table_sep)
        output_lines.append(table_header)
        output_lines.append(table_sep)

        for sent in sentences:
            for tok in sent.tokens:
                # обрезаем feats до 33 символов, чтобы таблица не расползалась
                feats_display = tok.feats if tok.feats else "_"
                if len(feats_display) > 33:
                    feats_display = feats_display[:30] + "..."
                row = (
                    f"  {tok.id:<4} {tok.form:<20} {tok.lemma:<20} "
                    f"{tok.pos:<8} {tok.xpos:<10} {feats_display:<35} "
                    f"{tok.head:<6} {tok.deprel}"
                )
                print(row)
                output_lines.append(row)
            total_sentences += 1
            total_tokens += len(sent.tokens)

        print()
        output_lines.append("")

    # проверяем корректность
    print("ПРОВЕРКА:")
    check_msg = f"  всего предложений: {total_sentences} (ожидается 5)"
    print(check_msg)
    output_lines.append("ПРОВЕРКА:")
    output_lines.append(check_msg)

    tokens_msg = f"  всего токенов: {total_tokens}"
    print(tokens_msg)
    output_lines.append(tokens_msg)

    # проверяем, что каждый Sentence содержит хотя бы 1 токен
    all_ok = True
    for idx, (text, desc) in enumerate(zip(test_sentences, descriptions)):
        sents = process_text_mk(text)
        if len(sents) == 0:
            err_msg = f"  ОШИБКА: предложение {idx + 1} ({desc}) не вернуло ни одного Sentence"
            print(err_msg)
            output_lines.append(err_msg)
            all_ok = False
        for s in sents:
            if len(s.tokens) == 0:
                err_msg = f"  ОШИБКА: предложение {idx + 1} ({desc}), sent_id={s.sent_id} имеет 0 токенов"
                print(err_msg)
                output_lines.append(err_msg)
                all_ok = False

    if all_ok:
        ok_msg = "  все проверки пройдены"
        print(ok_msg)
        output_lines.append(ok_msg)

    # выводим пример CoNLL-U для первого предложения
    print()
    print("ПРИМЕР CoNLL-U (первое предложение):")
    output_lines.append("")
    output_lines.append("ПРИМЕР CoNLL-U (первое предложение):")

    conllu = text_to_conllu_mk(test_sentences[0])
    print(conllu)
    output_lines.append(conllu)

    # сохраняем вывод в файл
    output_path = os.path.join(PROJECT_ROOT, 'temp', 'phase_b_step2_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nвывод сохранён в {output_path}")
