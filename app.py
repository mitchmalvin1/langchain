import os
from langchain import  PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import openai
import pinecone

load_dotenv()
app = Flask(__name__)

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

embeddings = OpenAIEmbeddings()

vectorstore = Pinecone.from_existing_index("test-index", embeddings)

#open text file in read mode
text_file = open("./chat_summary.txt", "r")
 
#read whole file to a string
data = text_file.read()
 
#close file
text_file.close()
 
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "Analyze the following conversation and return me the talking style, accent and common slang words that Fabian's wife likes to use." + data + "Return your answer in the following format: Talking style : ... ,Common Slang words : ... "},
    ]
)

talkingStyleStr = response['choices'][0]['message']['content']
print(talkingStyleStr)


#  create a memory object to track the inputs/outputs and hold a conversation
conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

userPrompt = input('Please enter the additional prompt that you would like to be added to the LLM : ')

## Uncomment this and specify into the prompt template when asked

# You are acting as Fabian's wife. Analyze the given conversation in the context. I will talk to you as Fabian and you will reply me as Fabian's wife. Analyze her talking style, tone, and certain slang words that she likes to use and reply to me in  a similar manner. You are to reply to me only once and only as Fabian's wife. Do not complete the conversation more than once. If there is not enough information, try to infer from the context and reply to me on your own. Try to imitate the talking style provided below sparingly and only when it is appropriate to do so.

prompt_template = """Use the following pieces of context to answer the question at the end.""" + userPrompt +  """

This is the talking style of Fabian's wife :""" + talkingStyleStr + """

This is the context given:
{context}

Fabian: {question}
Fabian's wife :"""

PROMPT = PromptTemplate(
     template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0,
)

chat_history = []
qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(search_kwargs={'k': 3}), 
    chain_type = "stuff",
    combine_docs_chain_kwargs= chain_type_kwargs,
    verbose = True,
    memory = conversational_memory)


@app.route('/chat',methods =['POST'])
def chat_bot():
    data = request.get_json()
    if not data :
        return jsonify ({"Error": "No data provided"}), 400
    
    messages = data.get("message")
    result = qa({"question": messages})
    return jsonify({"Status" : "success", "Message" : result["answer"]})


#This API Endpoint will take in a form file, save it to a local directory, and upload
@app.route('/ingest',methods = ['POST'])
def ingest_data():

    if not os.path.exists('./ingestData'):
        os.makedirs('./ingestData')
        print(f"Folder '{'./ingestData'}' created.")
 
    file = request.files['file']
    indexName = request.args.get('index')

    filename = secure_filename(file.filename)
    file.save(os.path.join('./ingestData', filename))

    documents = []
    loader = TextLoader(file_path='./ingestData/' + filename, encoding = 'cp850')
    documents.extend(loader.load())

    # split the documents, create embeddings for them, 
    # and put them in a vectorstore  to do semantic search over them.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = text_splitter.split_documents(documents)
    Pinecone.from_documents(documents, embeddings, index_name=indexName)
    os.remove('./ingestData/' + filename)

    return 'File successfully uploaded.', 200

if __name__ == '__main__' : 
    app.run()
     






