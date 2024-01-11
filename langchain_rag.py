# starting off 
from datetime import datetime

# Print the formatted date and time
print(f'start: {datetime.now().strftime("%A %b %d %I:%M %p")}')

from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# import nltk
# nltk.download('punkt')

path = "./pdffiles/"
infile = "2023 Availability 07-28.pdf"

# Get elements
raw_pdf_elements = partition_pdf(
    filename=path + infile,
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=False,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# Create a dictionary to store counts of each type
category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Print the formatted date and time
print(f'loaded elements: {datetime.now().strftime("%A %b %d %I:%M %p")}')

# Unique_categories will have unique elements
unique_categories = set(category_counts.keys())
category_counts

class Element(BaseModel):
    type: str
    text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# Print the formatted date and time
print(f'catagorized elements: {datetime.now().strftime("%A %b %d %I:%M %p")}')

# Summaries 
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# use canned prompt to summarize tables 
from langchain import hub 
prompt=hub.pull("rlm/multi-vector-retriever-summarization")

# authorize openAI 3.5 Turbo - free key from databayes 
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

'''from openai import OpenAI
client = OpenAI(
  organization='org-VTqL63szWVCqMu8NINpqVH8u',
)
'''

# Summary chain
#model = ChatOpenAI(temperature=0, model="gpt-4")
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Apply to tables
tables = [i.text for i in table_elements if len(i.text) > 0]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

# Apply to texts
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

jnk=0

# Print the formatted date and time
print(f'summarized elements: {datetime.now().strftime("%A %b %d %I:%M %p")}')

# add to vector store 
'''Use Multi Vector Retriever with summaries:
https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector#summary 

InMemoryStore stores the raw text, tables
vectorstore stores the embedded summaries
https://python.langchain.com/docs/integrations/stores/

'''

import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# Print the formatted date and time
print(f'stored texrt and tables: {datetime.now().strftime("%A %b %d %I:%M %p")}')

from langchain_core.runnables import RunnablePassthrough

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

while True:
    # Prompt the user to enter a string
    user_input = input("Enter a string: ")
    print(chain.invoke(user_input)) 
    jnk=0
