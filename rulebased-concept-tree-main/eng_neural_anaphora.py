import os
import time
import socket
from stanza.server import CoreNLPClient
import stanza

stanza.install_corenlp()

# Укажи путь к установленному CoreNLP
os.environ["$CORENLP_HOME"] = "/Users/mac/stanza_corenlp/stanford-corenlp-4.5.8"


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class CoreferenceResolver:
    def __init__(self, endpoint="http://localhost:9000", memory="4G", algorithm="statistical"):
        self.endpoint = endpoint
        self.memory = memory
        self.algorithm = algorithm
        self.client = None

    def start_client(self):
        if self.client is not None:
            try:
                # Проверка, жив ли клиент
                self.client.annotate("Test", properties={"annotators": "tokenize", "outputFormat": "json"})
                return
            except Exception:
                self.client.stop()
                self.client = None

        properties = {
            "annotators": "tokenize,ssplit,pos,lemma,ner,parse,coref",
            "coref.algorithm": self.algorithm,
        }

        # Проверка, доступен ли порт
        port = int(self.endpoint.rsplit(":", 1)[-1])
        if is_port_in_use(port):
            new_port = get_free_port()
            print(f"[INFO] Port {port} in use, switching to available port {new_port}")
            port = new_port
            self.endpoint = f"http://localhost:{port}"

        self.client = CoreNLPClient(
            properties=properties,
            endpoint=self.endpoint,
            timeout=300000,
            memory=self.memory,
            be_quiet=True,
            max_char_length=100000,
            threads=4,
        )
        self.client.start()

    def stop_client(self):
        if self.client:
            self.client.stop()
            self.client = None

            # Подождать, пока порт освободится
            timeout = 5  # секунд
            interval = 0.2
            port = int(self.endpoint.rsplit(":", 1)[-1])
            waited = 0
            while is_port_in_use(port) and waited < timeout:
                time.sleep(interval)
                waited += interval

    def resolve_coreferences(self, text):
        if not self.client:
            raise RuntimeError("CoreNLPClient is not started.")
        document = self.client.annotate(text)
        replacements = []

        for chain in document.corefChain:
            representative = chain.mention[chain.representative]
            rep_sentence = document.sentence[representative.sentenceIndex]
            rep_tokens = rep_sentence.token[representative.beginIndex:representative.endIndex]
            rep_text = " ".join(t.word for t in rep_tokens)

            is_object = any(t.pos.startswith("NN") for t in rep_tokens)

            for mention in chain.mention:
                if mention.mentionID != representative.mentionID:
                    mention_sentence = document.sentence[mention.sentenceIndex]
                    mention_tokens = mention_sentence.token[mention.beginIndex:mention.endIndex]
                    is_possessive = any(
                        t.pos == "PRP$" or t.word.lower() in {"my", "your", "his", "her", "its", "our", "their"}
                        for t in mention_tokens
                    )
                    replacement_text = f"{rep_text}'s" if is_possessive and is_object else rep_text
                    replacements.append({
                        "start": mention_sentence.token[mention.beginIndex].beginChar,
                        "end": mention_sentence.token[mention.endIndex - 1].endChar,
                        "replacement": replacement_text,
                    })

        resolved = text
        for r in sorted(replacements, key=lambda x: x["start"], reverse=True):
            resolved = resolved[:r["start"]] + r["replacement"] + resolved[r["end"]:]
        return resolved


_resolver = None

def get_resolver():
    global _resolver
    if _resolver is None:
        _resolver = CoreferenceResolver()
        _resolver.start_client()
    return _resolver

def resolve_anaphora_en(text):
    resolver = get_resolver()
    return resolver.resolve_coreferences(text)

def shutdown_resolver():
    global _resolver
    if _resolver:
        _resolver.stop_client()
        _resolver = None


if __name__ == "__main__":
    text = "Alice went to the store. She bought some milk."
    resolved_text = resolve_anaphora_en(text)
    print("Resolved:", resolved_text)