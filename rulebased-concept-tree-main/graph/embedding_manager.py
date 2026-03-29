import os
import numpy as np
from numpy.typing import NDArray


class EmbeddingManager:
    # путь к бинарной FastText-модели для македонского (cc.mk.300.bin, ~4.5 ГБ)
    # модель хранится в temp/models/ и не коммитится в git
    _MODEL_DIR = os.path.join(
        os.path.dirname(__file__), '..', '..', 'temp', 'models'
    )
    _BIN_PATH = os.path.join(_MODEL_DIR, 'cc.mk.300.bin')
    # текстовый формат (.vec) — fallback, если бинарная модель недоступна
    _VEC_PATH = os.path.join(_MODEL_DIR, 'cc.mk.300.vec')

    # кеш модели: загружаем один раз, потом переиспользуем (lazy initialization)
    # без кеша модель загружалась бы при каждом вызове get_embedding() — это 30-60 секунд каждый раз
    _model = None
    # тип загруженной модели: 'fasttext' (полная, с subword) или 'gensim' (только словарь)
    _model_type = None
    # размерность эмбеддингов (300 для cc.mk.300)
    _dim = 300

    @classmethod
    def _load_model(cls) -> None:
        # lazy initialization: загружаем модель только при первом вызове get_embedding()
        # это экономит время и память, если эмбеддинги не нужны в конкретном запуске

        # вариант 1: бинарная FastText-модель (лучшее качество)
        # .bin-формат содержит subword-информацию, поэтому может выдать вектор
        # даже для слов, которых нет в словаре (OOV = out-of-vocabulary)
        # это критично для македонского с его богатой морфологией
        if os.path.isfile(cls._BIN_PATH):
            import fasttext
            # suppress warning about deprecated load_model
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            cls._model = fasttext.load_model(cls._BIN_PATH)
            cls._model_type = 'fasttext'
            cls._dim = cls._model.get_dimension()
            return

        # вариант 2: текстовый .vec-файл через gensim KeyedVectors
        # .vec содержит только готовые вектора без subword-информации
        # минус: OOV-слова получат нулевой вектор (нет subword-разложения)
        # плюс: файл ~1.5 ГБ вместо ~4.5 ГБ, быстрее загружается
        if os.path.isfile(cls._VEC_PATH):
            from gensim.models import KeyedVectors
            cls._model = KeyedVectors.load_word2vec_format(
                cls._VEC_PATH, binary=False
            )
            cls._model_type = 'gensim'
            cls._dim = cls._model.vector_size
            return

        # если ни один файл не найден — ошибка с инструкцией, куда положить модель
        raise FileNotFoundError(
            f"FastText-модель не найдена. Скачайте cc.mk.300.bin.gz из "
            f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mk.300.bin.gz , "
            f"распакуйте и положите в {cls._MODEL_DIR}"
        )

    @classmethod
    def get_embedding(cls, word: str) -> NDArray[np.float64]:
        # эмбеддинг (embedding) — числовой вектор, который кодирует смысл слова
        # похожие по смыслу слова имеют похожие вектора (малый cosine distance)

        # загружаем модель при первом вызове (lazy init)
        if cls._model is None:
            cls._load_model()

        if cls._model_type == 'fasttext':
            # FastText: get_word_vector() работает для любого слова,
            # даже если его нет в словаре — модель разбивает слово на части (subwords)
            # и складывает их вектора
            vec = cls._model.get_word_vector(word)
            return vec.astype(np.float64)

        if cls._model_type == 'gensim':
            # gensim KeyedVectors: нет subword-разложения,
            # поэтому для OOV-слов возвращаем нулевой вектор
            if word in cls._model:
                return cls._model[word].astype(np.float64)
            # OOV-слово: пробуем lowercase-вариант (в корпусе могут быть заглавные)
            lower = word.lower()
            if lower in cls._model:
                return cls._model[lower].astype(np.float64)
            # если слово совсем не найдено — нулевой вектор
            return np.zeros(cls._dim, dtype=np.float64)

        # на случай неизвестного типа модели (не должно случиться)
        return np.zeros(cls._dim, dtype=np.float64)

    @classmethod
    def get_dimension(cls) -> int:
        # возвращаем размерность эмбеддингов (300 для cc.mk.300)
        if cls._model is None:
            cls._load_model()
        return cls._dim

    @classmethod
    def reset(cls) -> None:
        # сброс кеша модели — полезно при тестировании или переключении моделей
        cls._model = None
        cls._model_type = None
        cls._dim = 300
