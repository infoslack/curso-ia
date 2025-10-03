import os
import nltk
import shutil
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]


def preprocess(text):
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    tokens = [word for word in tokens if word.isalnum()]
    stopwords = set(nltk.corpus.stopwords.words("portuguese")) - {"e", "ou", "não"}
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


text = "Machine learning é um campo da inteligência artificial. que permite que computadores aprendam padrões a partir de dados."
preprocess(text)

if os.path.exists("index_dir"):
    shutil.rmtree("index_dir")
os.mkdir("index_dir")

schema = Schema(title=ID(stored=True, unique=True), content=TEXT(stored=True))

index = create_in("index_dir", schema)

writer = index.writer()
for i, doc in enumerate(documents):
    writer.add_document(title=str(i), content=doc)
writer.commit()

query = "machine E learning"


def boolean_search(query, index):
    parser = QueryParser("content", schema=index.schema)
    parsed_query = parser.parse(query)

    with index.searcher() as searcher:
        results = searcher.search(parsed_query)
        return [(hit["title"], hit["content"]) for hit in results]


boolean_search(query, index)
