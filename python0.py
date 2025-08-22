from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pdfplumber

class One:
    def loader(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages = []
        for root, dirs, files in os.walk(self.pdf_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    self.pages.extend(docs)
        print(f"Loaded {len(self.pages)} pages from the PDF(s).")

    def extract_tables(self):
        self.table_chunks = []
        for root, dirs, files in os.walk(self.pdf_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, start=1):
                            tables = page.extract_tables()
                            for tbl_index, table in enumerate(tables, start=1):
                                # Replace None with "" in each row
                                cleaned_rows = [[cell if cell is not None else "" for cell in row] for row in table]
                                table_text = "\n".join([", ".join(row) for row in cleaned_rows])
                                self.table_chunks.append(
                                    Document(
                                        page_content=table_text,
                                        metadata={
                                            "type": "table",
                                            "page": page_num,
                                            "table_no": tbl_index,
                                            "source": file_path
                                        }
                                    )
                                )
        print(f"Extracted {len(self.table_chunks)} tables from all PDFs.")

    def splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.normal_chunk = self.splitter.split_documents(self.pages)
        print(f"Split into {len(self.normal_chunk)} chunks.")
        # Now, chunks is a list of Document objects
        self.chunk = self.normal_chunk + self.table_chunks
        print(f"Total chunks created: {len(self.chunk)}")

    def embed_into_vector_and_store(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorStore = FAISS.from_documents(self.chunk, self.embedding_model)
        print("Vectors are stored in FAISS.")

        # Yes — FAISS.from_documents(documents, embeddings) internally: 1️⃣ Converts chunks (documents) into vectors,  Creates a FAISS index and stores the vectors
    def save_vector(self):
        self.vectorStore.save_local("vector_store.faiss")
        print("Vector store saved locally as 'vector_store.faiss'.")

if __name__ == "__main__":
    pdf_path = "main_folder"  
    one_instance = One()

    one_instance.loader(pdf_path)
    one_instance.extract_tables()
    one_instance.splitter()
    one_instance.embed_into_vector_and_store()
    one_instance.save_vector()


