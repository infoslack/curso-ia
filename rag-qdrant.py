from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

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

client = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")
# qdrant = QdrantClient(":memory:")
qdrant = QdrantClient(path="db/data")

vector_size = model.get_sentence_embedding_dimension()

qdrant.create_collection(
    collection_name="ml_documents",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

points = []
for idx, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    points.append(PointStruct(id=idx, vector=embedding, payload={"text": doc}))

qdrant.upsert(collection_name="ml_documents", points=points, wait=True)


def retrieve(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    search_result = qdrant.query_points(
        collection_name="ml_documents",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return [(hit.payload["text"], hit.score) for hit in search_result.points]


def generate_answer(query, retrieve_docs):
    context = "\n".join([doc for doc, _ in retrieve_docs])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Você é um especialista em machine learning. Use apenas o contexto fornecido para responder as perguntas.",
            },
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def rag(query, top_k=3):
    retrieved = retrieve(query, top_k)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


answer, docs = rag("O que é machine learning?")
print(answer)
print(docs)

for doc, similarity in docs:
    print(f" - {similarity:.3f}: {doc}")
