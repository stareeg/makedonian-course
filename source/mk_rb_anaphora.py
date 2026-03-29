# mk_rb_anaphora.py
# модуль разрешения анафоры для македонского языка
# rule-based RAP (Resolution of Anaphora Procedure)
# заменяет местоимения (тој, таа, тоа, тие и др.) на их антецеденты

import sys
import os
import time
import logging

# принудительно ставим UTF-8, чтобы кириллица не ломалась в Windows-консоли
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# путь к корню проекта (на уровень выше папки source/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# путь к папке с модулем sent_class.py (классы Token и Sentence)
TREE_DIR = os.path.join(PROJECT_ROOT, 'rulebased-concept-tree-main')
# добавляем в sys.path, чтобы import sent_class работал
if TREE_DIR not in sys.path:
    sys.path.insert(0, TREE_DIR)

# импортируем классы Token и Sentence для работы с разобранными предложениями
from sent_class import Token, Sentence

# настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
# логгер для нашего модуля
logger = logging.getLogger('mk_rb_anaphora')

# клитические местоимения македонского языка
# их не резолвим: в македонском они обычно дублируют полное дополнение
# ("Го видов Марко" -- "го" дублирует "Марко", резолвить его нет смысла)
MK_CLITICS = {
    'ме', 'те', 'го', 'ја', 'не', 'ве', 'ги',  # аккузатив (краткие формы)
    'ми', 'ти', 'му', 'ѝ', 'ни', 'ви', 'им',    # датив (краткие формы)
    'се',                                          # возвратное местоимение
}

# местоимения, для которых ищем антецедент
# это полные (ударные) формы личных и указательных местоимений
MK_PRONOUNS_TO_RESOLVE = {
    'тој', 'таа', 'тоа', 'тие',       # именительный падеж (личные/указательные)
    'него', 'неа', 'нив',              # полные формы аккузатива (не клитики)
    'нему', 'нејзе', 'ним',            # полные формы датива (не клитики)
}


class NPExtractorMK:
    """
    Извлечение именных групп (NP) из предложения для македонского языка.
    Работает с UPOS-тегами (NOUN, PROPN, PRON) и deprel-связями из CoNLL-U.
    """

    def extract_nps(self, sent):
        """
        Принимает объект Sentence, возвращает список NP (словарей).
        Каждый NP -- именная группа с вершиной-существительным или именем собственным.
        """
        # сюда складываем найденные именные группы
        nps = []

        # перебираем все токены предложения
        for tok in sent.tokens:
            # берём только существительные (NOUN), имена собственные (PROPN) и местоимения (PRON)
            if tok.pos not in {'NOUN', 'PROPN', 'PRON'}:
                continue

            # собираем id всех токенов, входящих в эту именную группу
            np_tokens = self._collect_np_tokens(tok, sent)

            # создаём словарь с описанием NP
            np = {
                # id вершины именной группы (само существительное или имя)
                'head_id': tok.id,
                # множество id токенов, входящих в NP (отсортированное для красоты)
                'tokens': sorted(np_tokens),
                # лемма вершины (нормальная форма)
                'head_lemma': tok.lemma,
                # словоформа вершины (как в тексте)
                'head_form': tok.form,
                # UPOS-тег вершины
                'pos': tok.pos,
                # грамматический род (MASC, FEM, NEUT или UNK)
                'gender': self._get_gender(tok),
                # грамматическое число (SG, PL или UNK)
                'number': self._get_number(tok),
                # лицо (для существительных всегда 3)
                'person': 3,
                # синтаксическая роль в предложении (SUBJ, OBJ, IOBJ, OBL, OTHER)
                'gram_role': self._get_grammatical_role(tok),
                # номер предложения (для отслеживания расстояния в дискурсе)
                'sentence_id': sent.sent_id,
            }

            # добавляем NP в список
            nps.append(np)

        return nps

    def _collect_np_tokens(self, head, sent):
        """
        Собирает id всех модификаторов вершины NP.
        Берёт зависимые с deprel: amod (прилагательное), det (определитель), nummod (числительное).
        """
        # начинаем с самой вершины
        tokens = {head.id}

        # ищем зависимые токены, которые модифицируют вершину NP
        for tok in sent.tokens:
            # токен зависит от нашей вершины и имеет нужный тип связи
            if tok.head == head.id and tok.deprel in {'amod', 'det', 'nummod'}:
                # добавляем id модификатора в множество
                tokens.add(tok.id)

        return tokens

    def _get_grammatical_role(self, tok):
        """
        Определяет синтаксическую роль токена по его deprel.
        Маппинг: nsubj -> SUBJ, obj -> OBJ, iobj -> IOBJ, obl -> OBL.
        """
        # подлежащее (включая пассивное)
        if tok.deprel in {'nsubj', 'nsubj:pass'}:
            return 'SUBJ'
        # прямое дополнение
        if tok.deprel in {'obj', 'dobj'}:
            return 'OBJ'
        # косвенное дополнение
        if tok.deprel == 'iobj':
            return 'IOBJ'
        # обстоятельство (предложная группа)
        if tok.deprel == 'obl':
            return 'OBL'
        # всё остальное
        return 'OTHER'

    def _get_gender(self, tok):
        """
        Извлекает грамматический род из строки FEATS токена.
        CLASSLA для македонского заполняет Gender для существительных и местоимений.
        """
        # если feats пустой или прочерк, род неизвестен
        if not tok.feats or tok.feats == '_':
            return 'UNK'

        # перебираем пары ключ=значение в строке FEATS
        for pair in tok.feats.split('|'):
            # ищем пару Gender=...
            if pair.startswith('Gender='):
                # достаём значение после знака =
                value = pair.split('=')[1]
                # маппим на наши сокращения
                if value == 'Masc':
                    return 'MASC'
                if value == 'Fem':
                    return 'FEM'
                if value == 'Neut':
                    return 'NEUT'

        # если Gender не нашли в FEATS
        return 'UNK'

    def _get_number(self, tok):
        """
        Извлекает грамматическое число из строки FEATS токена.
        """
        # если feats пустой или прочерк, число неизвестно
        if not tok.feats or tok.feats == '_':
            return 'UNK'

        # перебираем пары ключ=значение в строке FEATS
        for pair in tok.feats.split('|'):
            # ищем пару Number=...
            if pair.startswith('Number='):
                # достаём значение после знака =
                value = pair.split('=')[1]
                # маппим на наши сокращения
                if value == 'Sing':
                    return 'SG'
                if value == 'Plur':
                    return 'PL'

        # если Number не нашли в FEATS
        return 'UNK'


class MorphFilterMK:
    """
    Морфологический фильтр для македонского языка.
    Отсеивает кандидатов, не совпадающих по роду и числу с местоимением.
    Все признаки берём из FEATS (строка CoNLL-U), без pymorphy (его для mk нет).
    """

    def filter(self, pronoun_token, candidates):
        """
        Принимает токен-местоимение и список кандидатов-NP.
        Возвращает только тех кандидатов, которые совпадают по роду и числу.
        """
        # извлекаем род местоимения из его FEATS
        pron_gender = self._extract_gender(pronoun_token.feats)
        # извлекаем число местоимения из его FEATS
        pron_number = self._extract_number(pronoun_token.feats)

        # если ни род, ни число не известны, фильтровать нечем
        if pron_gender == 'UNK' and pron_number == 'UNK':
            return candidates

        # сюда складываем подходящих кандидатов
        filtered = []

        for np in candidates:
            # проверяем совпадение числа (если у местоимения оно определено)
            if pron_number != 'UNK' and np.get('number', 'UNK') != 'UNK':
                # если число не совпадает, пропускаем кандидата
                if np['number'] != pron_number:
                    continue

            # проверяем совпадение рода (если у местоимения он определён)
            if pron_gender != 'UNK' and np.get('gender', 'UNK') != 'UNK':
                # если род не совпадает, пропускаем кандидата
                if np['gender'] != pron_gender:
                    continue

            # кандидат прошёл фильтр, добавляем
            filtered.append(np)

        return filtered

    def _extract_gender(self, feats_str):
        """
        Извлекает род из строки FEATS (формат CoNLL-U: "Gender=Masc|Number=Sing|...").
        CLASSLA для македонского всегда заполняет Gender для существительных и местоимений.
        """
        # если feats пустой или прочерк, род неизвестен
        if not feats_str or feats_str == '_':
            return 'UNK'

        # перебираем пары ключ=значение
        for pair in feats_str.split('|'):
            # ищем пару с Gender
            if pair.startswith('Gender='):
                # достаём значение после знака =
                value = pair.split('=')[1]
                # маппим стандартные значения UPOS
                if value == 'Masc':
                    return 'MASC'
                if value == 'Fem':
                    return 'FEM'
                if value == 'Neut':
                    return 'NEUT'

        # если Gender не нашли
        return 'UNK'

    def _extract_number(self, feats_str):
        """
        Извлекает число из строки FEATS (формат CoNLL-U: "Number=Sing|...").
        """
        # если feats пустой или прочерк, число неизвестно
        if not feats_str or feats_str == '_':
            return 'UNK'

        # перебираем пары ключ=значение
        for pair in feats_str.split('|'):
            # ищем пару с Number
            if pair.startswith('Number='):
                # достаём значение после знака =
                value = pair.split('=')[1]
                # маппим стандартные значения UPOS
                if value == 'Sing':
                    return 'SG'
                if value == 'Plur':
                    return 'PL'

        # если Number не нашли
        return 'UNK'


class BindingFilterMK:
    """
    Фильтр по Binding Theory (Condition B) для македонского языка.
    Condition B языконезависимо: местоимение не может быть кореферентно
    с NP, которая зависит от того же предиката в том же предложении.
    Логика переиспользована из английской версии без изменений.
    """

    def filter_coargument(self, pronoun_token, candidate_nps, sent):
        """
        Убирает кандидатов из того же предложения,
        у которых общий head (предикат) с местоимением.
        NP из предыдущих предложений не фильтруются (Condition B не действует).
        """
        # сюда складываем тех кандидатов, которые прошли фильтр
        filtered = []

        # находим вершину (предикат), от которой зависит местоимение
        pron_head = sent.get_token(pronoun_token.head)
        # если вершину не удалось найти, фильтровать нечем, возвращаем всех
        if not pron_head:
            return candidate_nps

        # проверяем каждого кандидата
        for np in candidate_nps:
            # NP из другого предложения: Condition B не применимо, оставляем
            if np['sentence_id'] != sent.sent_id:
                filtered.append(np)
                continue

            # ищем вершину (head) кандидата в текущем предложении
            cand_head_tok = sent.get_token(np['head_id'])
            # если токен кандидата не найден в предложении, на всякий случай оставляем
            if not cand_head_tok:
                filtered.append(np)
                continue

            # проверяем, зависят ли кандидат и местоимение от одного предиката
            # если оба зависят от одного head, это ко-аргументы одного предиката
            if cand_head_tok.head == pron_head.id and pronoun_token.head == pron_head.id:
                # ко-аргумент того же предиката, Condition B запрещает кореференцию
                continue

            # кандидат прошёл фильтр
            filtered.append(np)

        return filtered


class DiscourseModelMK:
    """
    Дискурсивная модель с затуханием для отслеживания NP-кандидатов.
    Хранит NP из последних N предложений. Для коротких юмористических текстов
    достаточно окна в 3 предложения (max_sent_distance=3).
    Логика переиспользована из английской версии.
    """

    def __init__(self, max_sent_distance=3):
        # максимальное расстояние в предложениях, на котором NP ещё считаются активными
        self.max_sent_distance = max_sent_distance
        # список активных сущностей: каждая -- словарь с полями np, salience, last_seen
        self.entities = []

    def add_nps(self, nps, current_sent_id):
        """
        Добавляет NP из текущего предложения в дискурсивную модель.
        Начальный salience = 0 (реальный вес посчитает SalienceScorer при резолюции).
        """
        for np in nps:
            # создаём запись для каждой NP
            entry = {
                # ссылка на словарь NP (head_id, tokens, gender, number и т.д.)
                'np': np,
                # начальный вес значимости (пересчитывается при decay)
                'salience': 0,
                # номер предложения, в котором NP встретилась последний раз
                'last_seen': current_sent_id,
            }
            # добавляем в список активных сущностей
            self.entities.append(entry)

    def decay(self, current_sent_id):
        """
        Применяет затухание на основе расстояния в предложениях.
        NP, которые дальше max_sent_distance предложений, удаляются из модели.
        Вес оставшихся уменьшается экспоненциально: salience / 2^distance.
        """
        # сюда складываем сущности, которые ещё не устарели
        new_entities = []

        for ent in self.entities:
            # считаем расстояние от текущего предложения до последнего упоминания NP
            distance = current_sent_id - ent['last_seen']

            # если NP слишком далеко, убираем из модели
            if distance > self.max_sent_distance:
                continue

            # уменьшаем вес: чем дальше NP, тем слабее
            ent['salience'] = ent['salience'] / (2 ** distance)
            # сущность ещё актуальна, оставляем
            new_entities.append(ent)

        # заменяем список активных сущностей на отфильтрованный
        self.entities = new_entities

    def get_candidates(self, current_sent_id):
        """
        Возвращает список NP-кандидатов, которые ещё находятся в окне дискурса.
        """
        # сюда складываем подходящих кандидатов
        candidates = []

        for ent in self.entities:
            # считаем расстояние от текущего предложения
            distance = current_sent_id - ent['last_seen']
            # берём только NP в пределах окна
            if distance <= self.max_sent_distance:
                candidates.append(ent['np'])

        return candidates


class SalienceScorerMK:
    """
    Ранжирование NP-кандидатов по значимости (salience) для RAP.
    Веса переиспользованы из английской версии: SUBJ=80, OBJ=50, IOBJ=40, OBL=40.
    """

    # веса для синтаксических ролей
    ROLE_WEIGHTS = {
        'SUBJ': 80,
        'OBJ': 50,
        'IOBJ': 40,
        'OBL': 40,
        'OTHER': 0,
    }

    # бонус за то, что NP упомянута в текущем предложении (а не в предыдущем)
    RECENCY_WEIGHT = 100
    # бонус за то, что NP -- самостоятельная вершина (не вложена в другую NP)
    HEAD_WEIGHT = 80
    # бонус за параллелизм: местоимение и антецедент играют одну роль (оба SUBJ и т.д.)
    PARALLELISM_BONUS = 35

    def score(self, pronoun_token, candidates, sent):
        """
        Для каждого кандидата считает вес и возвращает список пар (np, weight),
        отсортированный по убыванию веса. Лучший кандидат -- первый.
        """
        # сюда складываем пары (кандидат, его вес)
        scored = []

        for np in candidates:
            # начальный вес
            weight = 0

            # бонус за близость: NP из того же предложения ценнее
            if np['sentence_id'] == sent.sent_id:
                weight += self.RECENCY_WEIGHT

            # бонус за синтаксическую роль (подлежащее ценнее дополнения)
            weight += self.ROLE_WEIGHTS.get(np['gram_role'], 0)

            # бонус за "головную" NP: если NP состоит из одного токена, она простая
            if len(np['tokens']) == 1:
                weight += self.HEAD_WEIGHT

            # бонус за параллелизм: роль местоимения совпадает с ролью кандидата
            pron_role = self._get_pronoun_role(pronoun_token)
            if pron_role == np['gram_role']:
                weight += self.PARALLELISM_BONUS

            # добавляем пару (кандидат, его вес)
            scored.append((np, weight))

        # сортируем по убыванию веса: лучший кандидат первый
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_pronoun_role(self, pron_token):
        """
        Определяет синтаксическую роль местоимения по его deprel.
        Нужно для подсчёта бонуса за параллелизм.
        """
        # подлежащее (включая пассивное)
        if pron_token.deprel in {'nsubj', 'nsubj:pass'}:
            return 'SUBJ'
        # прямое дополнение
        if pron_token.deprel in {'obj', 'dobj'}:
            return 'OBJ'
        # косвенное дополнение
        if pron_token.deprel == 'iobj':
            return 'IOBJ'
        # обстоятельство (предложная группа)
        if pron_token.deprel == 'obl':
            return 'OBL'
        # всё остальное
        return 'OTHER'


def resolve_anaphora_mk(sentences):
    """
    Главная функция: разрешает анафору в списке предложений на македонском языке.

    Принимает: List[Sentence] (из prepare_mk.py, после обработки через CLASSLA + spaCy).
    Возвращает: тот же список Sentence, но с заменёнными формами местоимений.

    Алгоритм RAP (Resolution of Anaphora Procedure):
    1. Для каждого предложения извлекаем NP
    2. Затухание (decay) старых NP в дискурсивной модели
    3. Находим полные (не клитические) местоимения
    4. Для каждого местоимения:
       a. Берём кандидатов из discourse model (только из предыдущих предложений)
       b. Фильтруем по Binding Theory (Condition B)
       c. Фильтруем по роду и числу (MorphFilter)
       d. Ранжируем по значимости (SalienceScorer)
       e. Заменяем форму местоимения на лемму лучшего кандидата
    5. Добавляем NP текущего предложения в discourse model
    """
    # создаём экземпляры всех компонентов RAP
    np_extractor = NPExtractorMK()
    # морфологический фильтр (без pymorphy, всё из FEATS)
    morph_filter = MorphFilterMK()
    # фильтр по Binding Theory
    binding_filter = BindingFilterMK()
    # дискурсивная модель с окном в 3 предложения
    discourse_model = DiscourseModelMK(max_sent_distance=3)
    # ранжирование кандидатов по значимости
    salience_scorer = SalienceScorerMK()

    # счётчик замен для логирования
    total_replacements = 0

    # обрабатываем каждое предложение по порядку
    for sent in sentences:
        # 1. извлекаем именные группы из текущего предложения
        nps = np_extractor.extract_nps(sent)

        # 2. применяем затухание к NP из предыдущих предложений
        discourse_model.decay(sent.sent_id)

        # 3. находим местоимения для резолюции
        # берём только PRON, у которых лемма входит в список резолвируемых
        # и форма (в нижнем регистре) не входит в клитики
        pronouns = []
        for tok in sent.tokens:
            # проверяем UPOS: только местоимения
            if tok.pos != 'PRON':
                continue
            # проверяем, что форма не клитика (ме, те, го, ја и т.д.)
            if tok.form.lower() in MK_CLITICS:
                continue
            # проверяем, что лемма входит в список местоимений для резолюции
            if tok.lemma.lower() in MK_PRONOUNS_TO_RESOLVE or tok.form.lower() in MK_PRONOUNS_TO_RESOLVE:
                pronouns.append(tok)

        # 4. для каждого местоимения ищем антецедент
        for pron in pronouns:
            # 4a. берём кандидатов из discourse model
            all_candidates = discourse_model.get_candidates(sent.sent_id)

            # берём только кандидатов из предыдущих предложений (не из текущего)
            candidates = [
                np for np in all_candidates
                if np['sentence_id'] < sent.sent_id
            ]

            # если кандидатов нет, пропускаем местоимение
            if not candidates:
                continue

            # 4b. фильтр по Binding Theory (Condition B)
            candidates = binding_filter.filter_coargument(pron, candidates, sent)

            # если после фильтра кандидатов не осталось, пропускаем
            if not candidates:
                continue

            # 4c. фильтр по роду и числу
            morph_filtered = morph_filter.filter(pron, candidates)

            # если после морфологического фильтра кандидатов не осталось,
            # используем pre-filter список (fallback)
            # это бывает при сочинённых подлежащих: "Ана и Марија" (обе Sing),
            # а местоимение "тие" (Plur) — morph filter отсекает обоих
            if morph_filtered:
                candidates = morph_filtered

            # 4d. ранжируем кандидатов по значимости
            scored = salience_scorer.score(pron, candidates, sent)

            # если ранжирование вернуло пустой список, пропускаем
            if not scored:
                continue

            # 4e. берём лучшего кандидата (первый в отсортированном списке)
            best_np = scored[0][0]
            best_score = scored[0][1]

            # заменяем форму местоимения на лемму антецедента
            # в македонском падежная система упрощена, поэтому просто подставляем лемму
            old_form = pron.form
            pron.form = best_np['head_lemma']
            # обновляем лемму местоимения тоже
            pron.lemma = best_np['head_lemma']

            # логируем замену
            logger.debug(
                f"sent {sent.sent_id}: '{old_form}' -> '{pron.form}' "
                f"(score={best_score}, role={best_np['gram_role']})"
            )
            # увеличиваем счётчик замен
            total_replacements += 1

        # 5. обновляем текст предложения после всех замен
        # собираем текст из форм токенов
        sent.text = ' '.join(tok.form for tok in sent.tokens)

        # 6. добавляем NP текущего предложения в discourse model
        # (добавляем после резолюции, чтобы NP из текущего предложения
        #  не были кандидатами для местоимений в этом же предложении)
        discourse_model.add_nps(nps, sent.sent_id)

    # логируем итог
    logger.info(f"анафора: {total_replacements} замен в {len(sentences)} предложениях")

    # возвращаем тот же список (объекты изменены in-place)
    return sentences


if __name__ == '__main__':
    # тестовый блок: проверяем работу модуля на 3 македонских примерах

    # добавляем папку source в sys.path для импорта prepare_mk
    SOURCE_DIR = os.path.join(PROJECT_ROOT, 'source')
    if SOURCE_DIR not in sys.path:
        sys.path.insert(0, SOURCE_DIR)

    # импортируем функцию обработки текста из prepare_mk
    from prepare_mk import process_text_mk

    print()
    print("ТЕСТ mk_rb_anaphora.py: разрешение анафоры для македонского языка")
    print()

    # 3 тестовых текста (каждый -- два предложения, во втором есть местоимение)
    test_cases = [
        {
            'text': 'Марко отиде на пазар. Тој купи јаболка.',
            'description': 'тој -> Марко (мужской род, ед. число)',
            'expected': 'Марко',
        },
        {
            'text': 'Кучето трча низ дворот. Тоа беше среќно.',
            'description': 'тоа -> куче (средний род, ед. число)',
            'expected': 'куче',
        },
        {
            'text': 'Ана и Марија отидоа во градот. Тие купија облека.',
            'description': 'тие -> Ана или Марија (мн. число)',
            'expected': 'Ана/Марија',
        },
    ]

    # сюда будем собирать весь вывод для записи в файл
    output_lines = []
    output_lines.append('mk_rb_anaphora.py: тестовый запуск')
    output_lines.append(f'дата: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    output_lines.append('')

    for idx, case in enumerate(test_cases):
        header = f'ТЕСТ {idx + 1}: {case["description"]}'
        print(header)
        print(f'  вход: {case["text"]}')
        output_lines.append(header)
        output_lines.append(f'  вход: {case["text"]}')

        # шаг 1: обработка текста через гибридный pipeline (CLASSLA + spaCy)
        sentences = process_text_mk(case['text'])

        # выводим токены до резолюции
        output_lines.append('  токены до резолюции:')
        print('  токены до резолюции:')
        for sent in sentences:
            for tok in sent.tokens:
                # формируем строку с основными полями токена
                tok_line = (
                    f'    [{tok.id}] {tok.form:<15} lemma={tok.lemma:<15} '
                    f'pos={tok.pos:<6} deprel={tok.deprel:<10} feats={tok.feats}'
                )
                print(tok_line)
                output_lines.append(tok_line)

        # шаг 2: резолюция анафоры
        resolved = resolve_anaphora_mk(sentences)

        # выводим результат после резолюции
        result_texts = []
        for sent in resolved:
            # собираем текст из форм токенов
            sent_text = ' '.join(tok.form for tok in sent.tokens)
            result_texts.append(sent_text)

        result_str = ' '.join(result_texts)
        print(f'  выход: {result_str}')
        print(f'  ожидание: местоимение заменено на "{case["expected"]}"')
        output_lines.append(f'  выход: {result_str}')
        output_lines.append(f'  ожидание: местоимение заменено на "{case["expected"]}"')

        print()
        output_lines.append('')

    # итог
    summary = 'резолюция анафоры завершена, результаты выше'
    print(summary)
    output_lines.append(summary)

    # сохраняем вывод в файл
    output_path = os.path.join(PROJECT_ROOT, 'temp', 'phase_d_step7_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f'\nвывод сохранён в {output_path}')
