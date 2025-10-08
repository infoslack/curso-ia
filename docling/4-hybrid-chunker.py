from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300

converter = DocumentConverter()

result = converter.convert("./2408.09869v5.pdf")
document = result.document

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL), max_tokens=MAX_TOKENS
)

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)

chunks = list(chunker.chunk(document))
len(chunks)

for i, chunk in enumerate(chunks):
    print(f"==={i}===\n")
    txt_tokens = tokenizer.count_tokens(chunk.text)
    print(f"chunk.text ({txt_tokens} tokens):\n{chunk.text!r}")

print(chunks[4].meta.doc_items[0].prov[0].page_no)
print(chunks[4].meta.headings)
