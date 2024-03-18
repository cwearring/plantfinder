import fitz
from rank_bm25 import BM250kapi 
import pandas as pd
import numpy as np



doc=fitz.open('/Users/cwearring/code/Garden Centre Ordering Forms/Brookdale/BTN PDF Availability, July 13, 2023.pdf')

tbls = doc[0].find_tables()

jnk=0