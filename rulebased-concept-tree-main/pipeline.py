import prepare
from run_maltparser import annotate
import os
from make_graph_en import build_en_graph
from make_graph_ru import build_ru_graph
from graph.graph import visualize_graph, visualize_graph_interactive
from eng_rb_anaphora import resolve_anaphora_en
from ru_rb_anaphora import resolve_anaphora_ru
import nltk
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    # раскомментировать при первом запуске
    # nltk.download("punkt_tab") 

    text = 'Мама зашла в комнату. Бабушка увидела собаку в комнате.'
    lang = 'ru'
    
    sentences = sent_tokenize(text)
    input_path = os.path.abspath("input.conll")

    # Разметка TreeTagger
    with open(input_path, "w") as f:
        for sent in sentences:
            f.write(f"# text = {sent}\n")
            conll_data = prepare.sentence_to_conllx(sent, lang=lang)
            f.write(conll_data)
            f.write("\n")

    # Разметка maltparser
    annotate("input.conll", "output.conll", lang=lang)

    # Разрешем анафору по синтаксической разметке
    if lang == 'en':
        resolved_sentences = resolve_anaphora_en("output.conll")
    elif lang == 'ru':
        resolved_sentences = resolve_anaphora_ru("output.conll")
        print("\nResolved sentences:")
        for sent in resolved_sentences:
            print(sent.text)

    # Строим граф
    if lang == 'en':
        g = build_en_graph(resolved_sentences)
    else:
        g = build_ru_graph(resolved_sentences)

    print("Graph edges:")
    for e in g.edges:
        print(f"{e.agent_1} --[{e.meaning}]--> {e.agent_2}")

    visualize_graph_interactive(g, output="graph.html")