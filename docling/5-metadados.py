import langextract as lx
import textwrap
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("./2408.09869v5.pdf")
markdown_output = result.document.export_to_markdown()

first_pages = markdown_output[:6000]

prompt = textwrap.dedent("""\
Extract metadata from this technical report including title, all authors, 
affiliation, version number, and GitHub repository URLs.
Use exact text from the document.
""")

examples = [
    lx.data.ExampleData(
        text="Docling Technical Report\nVersion 1.0\nChristoph Auer Maksym Lysak Ahmed Nassar\nAI4K Group, IBM Research\nRüschlikon, Switzerland\ngithub.com/DS4SD/docling",
        extractions=[
            lx.data.Extraction(
                extraction_class="title",
                extraction_text="Docling Technical Report",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="author",
                extraction_text="Christoph Auer",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="author", extraction_text="Maksym Lysak", attributes={}
            ),
            lx.data.Extraction(
                extraction_class="affiliation",
                extraction_text="AI4K Group, IBM Research",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="url",
                extraction_text="github.com/DS4SD/docling",
                attributes={"type": "repository"},
            ),
        ],
    )
]

extraction_result = lx.extract(
    text_or_documents=first_pages,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o-mini",
)

lx.io.save_annotated_documents(
    [extraction_result], output_name="docling_paper_metadata.jsonl"
)

print("-" * 80)
for extraction in extraction_result.extractions:
    print(f"{extraction.extraction_class}: {extraction.extraction_text}")
    if extraction.attributes:
        print(f"  Atributos: {extraction.attributes}")
