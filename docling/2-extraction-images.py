import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0
pipeline_options.generate_picture_images = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

result = converter.convert("./2408.09869v5.pdf")

os.makedirs("images", exist_ok=True)

picture_counter = 0
for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        picture_counter += 1
        with open(f"images/picture_{picture_counter}.png", "wb") as fp:
            element.get_image(result.document).save(fp, "PNG")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)
