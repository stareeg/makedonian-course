import subprocess
import os
import shutil
import tempfile

# Путь к директории с maltparser (где лежат jar и .mco)
MALT_DIR = "/Users/mac/Desktop/vscode/gromov/maltparser"
JAR_NAME = "maltparser-1.9.2.jar"
JAR_PATH = os.path.join(MALT_DIR, JAR_NAME)

def annotate(input_path, output_path=None, lang="en"):
    """
    Запускает MaltParser на указанном входном .conll-файле.
    Логика:
      - если input_path уже в MALT_DIR и называется 'input.conll', запускаем прямо;
      - иначе копируем input_path -> MALT_DIR/input.conll, запускаем в cwd=MALT_DIR,
        после чего читаем output_path и возвращаем строку.
    Возвращает текст результата (CoNLL) или выбрасывает исключение subprocess.CalledProcessError.
    """
    if lang == "ru":
        MODEL_NAME = "russian"
    else:
        MODEL_NAME = "engmalt.linear-1.7"

    if not os.path.exists(JAR_PATH):
        raise FileNotFoundError(f"Cannot find jar at {JAR_PATH}. Check MALT_DIR.")

    abs_input = os.path.abspath(input_path)

    malt_input_name = "input.conll"
    malt_input = os.path.join(MALT_DIR, malt_input_name)

    if not (os.path.abspath(os.path.dirname(abs_input)) == os.path.abspath(MALT_DIR) and os.path.basename(abs_input) == malt_input_name):
        shutil.copy2(abs_input, malt_input)
    else:
        malt_input = abs_input

    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output.conll")
    else:
        output_path = os.path.abspath(output_path)

    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "java", "-Xmx1024m", "-jar", JAR_PATH,
        "-c", MODEL_NAME,
        "-i", malt_input,
        "-o", output_path,
        "-m", "parse"
    ]

    # Запускаем в рабочей директории MALT_DIR
    subprocess.run(cmd, check=True, cwd=MALT_DIR)

    # Читаем результат
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"MaltParser finished but did not produce {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        parsed = f.read()

    if malt_input != abs_input and os.path.exists(malt_input):
        try:
            os.remove(malt_input)
        except Exception:
            pass

    return parsed

if __name__ == "__main__":
    input_file = "input.conll"
    text = annotate(input_file, lang="en")
    print(text)