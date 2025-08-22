from python0 import One
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate


class Two:   
    def Extract_from_vectors(self, vector_store_path="vector_store.faiss"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded vector store from {vector_store_path}")

    def user_input(self):
        self.question = input("Ask: ")
        if self.question.lower() == 'exit':
            return None
        return self.question

    def input_guardrail(self, question):
        self.blocked_keywords = ["password", "ssn", "bypass", "ignore safety", "off-topic"]
        for word in self.blocked_keywords:
            if word.lower() in question.lower():
                print(f"Input Rejected ❌ -> Blocked due to input guard violation: {word}")
                return False, f"Blocked due to input guard violation: {word}"
        return True, "Input allowed"

    def retriever(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        # when a user asked question based on user question it makes top 5 chunks
        return self.retriever

    def prompting(self, question):
        # Later, when retriever.invoke(question) is called, the user question is embedded and used to pick the top 5 most semantically similar chunks from your vector DB.
        self.relevant_docs = self.retriever.invoke(question)
        
        all_texts = [doc.page_content for doc in self.relevant_docs]
        self.context = "\n".join(all_texts)

        # Build prompt with ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question using ONLY the provided context. "
                       "Do NOT use outside knowledge. If the answer is not in the context, reply: 'Not in document.'"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
        ])
        self.prompt = prompt_template.format(context=self.context, question=question)
        return self.prompt

    def model_calling(self):
        self.llm = OllamaLLM(model="llama3.2:latest")

    def get_response(self, prompt):
        self.response = self.llm.invoke(prompt)
        return self.response

    def output_guardrail(self, response):
        self.blocked_phrases = ["unsafe", "irrelevant", "off-topic"]
        for phrase in self.blocked_phrases:
            if phrase.lower() in response.lower():
                print(f"Output Rejected ❌ -> Blocked due to output guardrail violation: {phrase}")
                return False, f"Blocked due to output guardrail violation: {phrase}"
        return True, "Output allowed"
       

if __name__ == "__main__":
    two_instance = Two()
    two_instance.Extract_from_vectors()
    two_instance.retriever()
    print("Vector store and retriever are ready.")
    two_instance.model_calling()
    print("LLM model is ready for use.")
    while True:
        user_question = two_instance.user_input()
        if user_question is None:
            break
        allowed, msg = two_instance.input_guardrail(user_question)
        if allowed:
            print("✅ Input passed guardrails.")
        else:
            print(msg)
            continue
        
        prompt = two_instance.prompting(user_question)
        
        response = two_instance.get_response(prompt)
        out_allowed, out_msg = two_instance.output_guardrail(response)
        if out_allowed:
            print(f"Answer: {response}")
        else:
            print(out_msg)
            continue
