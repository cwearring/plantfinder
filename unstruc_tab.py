from unstructured.partition.pdf import partition_pdf

from transformers import logging
logging.set_verbosity_error()

filename = "./pdffiles/2023 Availability 07-28.pdf"

elements = partition_pdf(filename=filename, infer_table_structure=True)
tables = [el for el in elements if el.category == "Table"]

for n,t in enumerate(tables): 
    with open(f"./pdffiles/table_{n}.html", "w") as file1:
        # Writing data to a file
        file1.write(t.metadata.text_as_html)

print(tables[0].text)
print(tables[0].metadata.text_as_html)

