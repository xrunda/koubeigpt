# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader

class LineTextSplitter:
    def __init__(self):
        pass

    def split_documents(self, documents):
        chunks = []
        for document in documents:
            lines = document.split('\n')
            chunks.extend(lines)
        return chunks


# load the document and split it into chunks
loader = CSVLoader("reshaped_car_data_1k.csv")
documents = loader.load()


# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
# db = Chroma.from_documents(docs, embedding_function)

# query it
query = "本田奥德赛2022款 2.0L e:HEV 锐·耀享福祉版售价,35.48万"
# docs = db.similarity_search(query)

# save to disk
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# docs = db2.similarity_search(query)

# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search('卡罗拉油耗')
print(db3)
print(docs)