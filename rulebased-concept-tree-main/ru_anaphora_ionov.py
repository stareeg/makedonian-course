from russian_anaphora_resolver.anaphora import resolve_anaphora
import pymorphy3


morph = pymorphy3.MorphAnalyzer()


def safe_inflect(antecedent: str, pronoun: str) -> str:
    ant = morph.parse(antecedent)[0]
    pro = morph.parse(pronoun)[0]

    grammemes = set()

    for g in ("case", "number", "gender"):
        val = getattr(pro.tag, g, None)
        if val:
            grammemes.add(val)

    # pymorphy3 требует строки
    grammemes = set(map(str, grammemes))

    inflected = ant.inflect(grammemes)

    if inflected:
        return inflected.word

    return antecedent

def resolve_anaphora_ru(text: str) -> str:
    """
    Applies rule-based anaphora resolution to Russian text
    and returns the text with pronouns replaced by their antecedents.

    This is a thin wrapper around the rule-based resolver (Ionov).
    """

    links = resolve_anaphora(text)

    links = sorted(
        links,
        key=lambda x: x["anaphora_offset"],
        reverse=True,
    )

    resolved_text = text

    for link in links:
        a_off = link["anaphora_offset"]
        a_len = link["anaphora_length"]
        antecedent = link["antecedent_token"]
        pronoun = resolved_text[a_off : a_off + a_len]

        replacement = safe_inflect(antecedent, pronoun)

        resolved_text = (
            resolved_text[:a_off]
            + replacement
            + resolved_text[a_off + a_len :]
        )

    return resolved_text


if __name__ == "__main__":
    # text = "Мама была уставшей. Она зашла в комнату."
    # text = 'Петя увидел Машу, которая вышла гулять.'
    # text = 'Мама зашла в комнату. Она начала готовить ужин.'
    # text = 'Аня нашла свой кошелёк.'
    text = 'Петя увидел Машу. Она улыбнулась ему.'
    print(resolve_anaphora_ru(text))