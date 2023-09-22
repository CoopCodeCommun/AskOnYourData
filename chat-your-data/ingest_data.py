
import pickle
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# utiliser https://python.langchain.com/en/latest/ecosystem/unstructured.html ?
loader = TextLoader("tibillet_all_text.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
docsearch = Chroma.from_documents(documents, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())

# Save vectorstore
with open("vectorstore_tibillet.pkl", "wb") as f:
    pickle.dump(vectorstore, f)



# # Load Data
# loader = UnstructuredFileLoader("tibillet_all_text.txt")
# raw_documents = loader.load()

# # Split text
# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(raw_documents)


# # Load Data to vectorstore
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents, embeddings)


