import os
from langchain import  PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import pinecone
import boto3

load_dotenv()
app = Flask(__name__)


 
chain_instances = {}
conversational_memory_instances ={}

embeddings = OpenAIEmbeddings()
s3_client = boto3.client('s3')
session = boto3.Session()
print(session.get_credentials().get_frozen_credentials().access_key)
print(session.get_credentials().get_frozen_credentials().secret_key)

pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
)
possible_NER_keys = ['influencer','user','entities','talking-style','accent', 'slang']


def retrieve_ingestion_template(influencer) :
    NER_map ={}
    currKey = ""
    keyIndex = 0

    if checkPath(influencer,'digital-immortality-ingestion-template') : 
        response = s3_client.get_object(Bucket='digital-immortality-ingestion-template', Key=f'{influencer}')
       
         # The 'Body' attribute contains the content of the file
        file_content = response['Body'].read().decode('utf-8')
    
        lines = file_content.splitlines()

        for line in lines:
            if keyIndex < len(possible_NER_keys) and line.startswith(possible_NER_keys[keyIndex]) :
                currKey = possible_NER_keys[keyIndex]
                NER_map[currKey] = ""
                keyIndex +=1
            elif len(line) > 0:
                NER_map[currKey] += line

        prompt_template = f"""Use the following pieces of context to answer the question at the end.

        You are acting as {NER_map['influencer']}. Analyze the given conversation in the context. I will talk to you as {NER_map['user']} and you will reply me as {NER_map['influencer']}. Analyze {NER_map['influencer']}'s talking style, tone, and certain slang words that she likes to use and reply to me in  a similar manner. You are to reply to me only once and only as {NER_map['influencer']}. Do not complete the conversation more than once. If there is not enough information in the context provided, do not use the context at all and reply to me on your own. Try to imitate the talking style provided below sparingly and only when it is appropriate to do so. 

        These are all entities present : 
        {NER_map['entities']}

        This is the talking style of {NER_map['influencer']} : 

        Talking style: {NER_map['talking-style']}

        Accent: {NER_map['accent']}

        Common Slang words: {NER_map['slang']}


        This is the context given:""" + """
        {context}

        """+ f"{NER_map['user']}:" + """
        {question}
        """ + f"""{NER_map['influencer']} :
        """
        
        return prompt_template
            
    

def checkPath(file_path,bucket_name):
  result = s3_client.list_objects(Bucket=bucket_name, Prefix=file_path )
  exists=False
  if 'Contents' in result:
      exists=True
  return exists

def retrieve_history(user, influencer, conversational_memory_instance):
    filePath = f'chat-history/{influencer}/{user}'

    if checkPath(filePath,'digital-immortality-chat-history') : 
        response = s3_client.get_object(Bucket='digital-immortality-chat-history', Key=f'chat-history/{influencer}/{user}')
       
         # The 'Body' attribute contains the content of the file
        file_content = response['Body'].read().decode('utf-8')
    
        lines = file_content.splitlines()

        for line in lines:
            line_without_prefix = line.replace('Human:', '').replace('AI:', '')

            if line.startswith('Human:'):
                conversational_memory_instance.chat_memory.add_user_message(line_without_prefix)
    
            elif line.startswith('AI:'):
                conversational_memory_instance.chat_memory.add_ai_message(line_without_prefix)
        print(f'History retrieved for {user} and {influencer}')
        print(conversational_memory_instance.load_memory_variables({}))
        print()
    else :
        print('No history detected.')
        print()

condense_prompt = """Given the following conversation and a follow up input, rephrase the follow up input to be a standalone question, in its original language. If it is not related, do not do anything to the follow up input and simply return the follow up input as it is (without modifying it at all). If there is no conversation found, also just return the follow up input in its original form.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


def initialize_chat(userName, influencer) :
    global chain_instances
    global conversational_memory_instances

    # s3_client.upload_file('./fabian_chat.txt', 'digital-immortality-chat-history', 'chat-history/history')
    #Create an index using the influencer name
    indexName = 'test-index'
    if indexName not in pinecone.list_indexes():
        print("Index does not exist, creating a new index called " + indexName)
        pinecone.create_index(
            name=indexName, 
            metric="cosine",
            dimension=1536)

    vectorstore = Pinecone.from_existing_index(indexName, embeddings, namespace=influencer)

    #  create a memory object to track the inputs/outputs and hold a conversation
    # conversational_memory_instance = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # retrieve_history(userName, influencer,conversational_memory_instance)
    prompt_template = retrieve_ingestion_template(influencer)

    PROMPT = PromptTemplate(
     template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.5,
    )

    current_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={'k': 3}), 
        chain_type = "stuff",
        combine_docs_chain_kwargs= chain_type_kwargs,
        condense_question_prompt=PromptTemplate.from_template(""),
        condense_question_llm=None,
        verbose = True)
    
    chain_instances[(userName, influencer)] = current_chain
    # conversational_memory_instances[(userName, influencer)] = conversational_memory_instance
    print(f'Chain for {userName} and {influencer} formed')
    print('Here are all the current chains so far :')
    print(list(chain_instances.keys()))

    


@app.route('/startchat', methods = ['POST'])
def start_chat() :
    userName = request.args.get('user')
    influencer = request.args.get('influencer')
    initialize_chat(userName, influencer)

    return f'Chat bot initialized for {userName} and {influencer}', 200


@app.route('/chat',methods =['POST'])
def chat_bot():
    data = request.get_json()
    if not data :
        return jsonify ({"Error": "No data provided"}), 400
    
    userName = request.args.get('user')
    influencer = request.args.get('influencer')
    
    messages = data.get("message")
    result = chain_instances[(userName, influencer)]({"question": messages, "chat_history" :""})
    # print(conversational_memory_instances[(userName, influencer)].load_memory_variables({}))
    
    return jsonify({"Status" : "success", "Message" : result["answer"]})


#This API Endpoint will take in a form file, save it to a local directory, and upload
@app.route('/ingest',methods = ['POST'])
def ingest_data():

    if not os.path.exists('./ingestData'):
        os.makedirs('./ingestData')
        print(f"Folder '{'./ingestData'}' created.")
 
    file = request.files['file']
    indexName = request.args.get('index')
    influencer = request.args.get('influencer')
    print()

    filename = secure_filename(file.filename)
    file.save(os.path.join('./ingestData', filename))

    documents = []
    loader = TextLoader(file_path='./ingestData/' + filename, encoding = 'cp850')
    documents.extend(loader.load())

    # split the documents, create embeddings for them, 
    # and put them in a vectorstore  to do semantic search over them.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = text_splitter.split_documents(documents)
    Pinecone.from_documents(documents, embeddings, index_name=indexName, namespace = influencer )
    os.remove('./ingestData/' + filename)

    return 'File successfully uploaded.', 200

# Takes in an ingestion template and upload it into local file ./ingestData/{fileName}
# Upload the local data onto S3 bucket/influencer
@app.route('/ingest/template',methods = ['POST']) 
def ingest_template() :

    if not os.path.exists('./ingestData'):
        os.makedirs('./ingestData')
        print(f"Folder '{'./ingestData'}' created.")

    file = request.files['file']
    influencer = request.args.get('influencer')

    filename = secure_filename(file.filename)
    file.save(os.path.join('./ingestData', filename))

    s3_client.upload_file(f'./ingestData/{filename}', 'digital-immortality-ingestion-template', f'{influencer}')   
    os.remove('./ingestData/' + filename)
    return 'Success',200

# Write the chat history into a local file ./historyData/history.txt and then upload into
# S3 bucket chat-history/{influencer}/{user}
@app.route('/save',methods = ['POST'])
def save_history() :
    userName = request.args.get('user')
    influencer = request.args.get('influencer')

    if not os.path.exists('./historyData'):
        os.makedirs('./historyData')
        print(f"Folder '{'./historyData'}' created.")

    #save to local history then upload to aws
    with open('./historyData/history.txt', "w") as file:
        chat_history=conversational_memory_instances[((userName, influencer))].load_memory_variables({})['chat_history']
        for message in chat_history:
            if hasattr(message, 'content'):
                if 'Human' in str(message.__class__):
                    file.write(str('Human:' + message.content))
                elif 'AI' in str(message.__class__):
                    file.write(str('AI:' + message.content))

    s3_client.upload_file('./historyData/history.txt', 'digital-immortality-chat-history', f'chat-history/{influencer}/{userName}')   
    os.remove('./historyData/history.txt') 
    return 'Success',200

if __name__ == '__main__' : 
    app.run()
     





