
from langchain import ConversationChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory



OPENAI_API_KEY = "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"

loader = TextLoader("data.txt")
documents = loader.load()

# split the documents, create embeddings for them, 
# and put them in a vectorstore  to do semantic search over them.
text_splitter = CharacterTextSplitter(chunk_size=60000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(documents, embeddings)

#  create a memory object
#  to track the inputs/outputs and hold a conversation
conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

custom_memory = ConversationBufferMemory(ai_prefix = "Kevin", human_prefix = "Mitch")


## I plan to fetch this from the DB in the future
## Use COnversationChain, comment this chunk out when trying with ConversationalRetrievalChain below
context = """
 Mitch is a second year Computer Engineering undergraduate who likes to play mobile games. He is from Indonesia and is the youngest son of his family. He came from a small city in North Sumatra called Medan. He has an older sister and is currently living alone in Singapore to pursue his studies.                                                                                                                                    """                                                                                                                                                                                                                                                                                                             
# template = """The following is a friendly conversation between a Mitch and Kevin. You will act as Kevin and reply me in a Singaporean or heavy Singlish accent and I will act as Mitch. The history of conversation between Mitch and Kevin is provided below.""" + context + """
# Current conversation:
# {history}
# Mitch: {input}
# Kevin:"""


# PROMPT = PromptTemplate(
#     input_variables=["history","input"], 
#     template=template
# )

# conversation = ConversationChain(
#     llm= ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
#     ),
#     prompt=PROMPT,
#     verbose=True, 
#     memory=custom_memory,
# )

# while True :
#      userInput = input('User:')
#      print(conversation.predict(input=userInput))



### Using Conversational Chain to fetch information from context 

prompt_template = """Use the following pieces of context to answer the question at the end. Analyze the given conversation in the context and understand the talking style as well as certain slang words that Wei often uses. You are to reply me as Wei and only Wei and follow his talking style, and I will act as Dimas. I will talk as Dimas, and then you reply once as Wei.

{context}

Dimas: {question}
Wei:"""
PROMPT = PromptTemplate(
     template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0,
)


qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(), 
    chain_type = "stuff",
    combine_docs_chain_kwargs= chain_type_kwargs,
    memory=conversational_memory)

while True :
     userInput = input('You (Dimas):')
     result = qa({"question": userInput})
     print(result["answer"])

