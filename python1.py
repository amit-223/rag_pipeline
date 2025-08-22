from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# import camelot
import pdfplumber
from langchain.schema import Document

class One:
    def loader(self, pdf_path):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(self.pdf_path)
        self.pages = self.loader.load()
        print(f"Loaded {len(self.pages)} pages from {self.pdf_path}")

    def splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.normal_chunk = self.splitter.split_documents(self.pages)
        print(f"Split into {len(self.normal_chunk)} chunks.")
        # Now, chunks is a list of Document objects

    def extract_tables(self):
        """Extract tables as separate chunks using pdfplumber"""
        self.table_chunks = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for tbl_index, table in enumerate(tables, start=1):
                    # Convert table into a readable string
                    table_text = "\n".join([", ".join(row) for row in table])
                    self.table_chunks.append(
                        Document(
                            page_content=table_text,
                            metadata={"type": "table", "page": page_num, "table_no": tbl_index}
                        )
                    )
        print(f"Extracted {len(self.table_chunks)} tables from PDF.")
        
        self.chunk = self.normal_chunk + self.table_chunks
        print(f"Total {len(self.chunk)} chunks (including {len(self.table_chunks)} table chunks).")

    def embed_into_vector_and_store(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorStore = FAISS.from_documents(self.chunk, self.embedding_model)
        print("Vectors are stored in FAISS.")

        # Yes — FAISS.from_documents(documents, embeddings) internally: 1️⃣ Converts chunks (documents) into vectors,  Creates a FAISS index and stores the vectors
    def save_vector(self):
        self.vectorStore.save_local("vector_store.faiss")
        print("Vector store saved locally as 'vector_store.faiss'.")

if __name__ == "__main__":
    pdf_path = "FTP2023_Chapter01.pdf"  # Replace with your PDF file path
    one_instance = One()

    one_instance.loader(pdf_path)
    one_instance.splitter()
    one_instance.extract_tables()
    one_instance.embed_into_vector_and_store()
    one_instance.save_vector()



    
   
    # self.Extract_plain_text = []
        # for i in self.chunk:
        #     self.Extract_plain_text.append(i.page_content) #Extract plain text using page_content
        # self.vectors = self.embedding_model.embed_documents(self.Extract_plain_text)
        # print(f"Generated {len(self.vectors)} vectors from the text chunks.")
    # def vectorstore_into_faiss(self):

        # self.embedding_dim = len(self.vectors[0])
        # self.index = faiss.IndexFlatL2(self.embedding_dim)
        # self.vector_store = FAISS(
        #     embedding_function=self.embedding_model,
        #     index=self.index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},
        # )
        # self.vector_store.add_texts(self.Extract_plain_text, self.vectors)
        # print("Vector store is ready with FAISS.")