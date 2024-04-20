#  ----------RAG Implementation-----------'

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = DirectoryLoader(
    r"data2")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


template = """
INSTRUCTIONS:

-You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 


<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:

"""

model = ChatOllama(model="mistral")
prompt_template = PromptTemplate(input_variables=["history", "context", "question"],
                                 template=template)
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever,  # Use the instance variable here
    chain_type_kwargs={"verbose": False, "prompt": prompt_template,
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
)


print("Chat Assistant \n")

# Initialize an empty list to store dishes
dishes = []

while True:
    print("\n******************************************************************")
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! Bye Bye.")
        break
    else:
        response_dict = chain.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
    print("******************************************************************\n")
