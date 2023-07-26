# langchain

## To initialize

Get the API key from OPEN AI's website and Pinecone and set it in your OS envrionment variable through the .env file

For instance : 

```
OPENAI_API_KEY=sk-****
PINECONE_API_KEY=***
PINECONE_ENV=us-****
```

Make sure you have `pip` installed in your device, it is just a package manager for python, then run

```
pip install -r requirements.txt
```


Copy all files (you can find in our group) and name them as follows :
```
chat_summary.txt : the file containing the small chunk from fabian's conversation used to generate the profile
```
Make sure you have the empty directory `/ingestData` in your folder so that the `ingest` API can put your files there

When prompted for the prompt, you can play around with it or just stick with : 
```
You are acting as Fabian's wife. Analyze the given conversation in the context. I will talk to you as Fabian and you will reply me as Fabian's wife. Analyze her talking style, tone, and certain slang words that she likes to use and reply to me in  a similar manner. You are to reply to me only once and only as Fabian's wife. Do not complete the conversation more than once. If there is not enough information, try to infer from the context and reply to me on your own. Try to imitate the talking style provided below sparingly and only when it is appropriate to do so.
```

Navigate to the correct directory and simply run :
```
flask run
```

You can comment out certain parts of the `print` statement in the code and remove the `verbose=True` property in the `ConversationalRetrievalChain` if you do not wish to see so mush logging that is cluttering your CLI.

API foot print : 

POST /chat 
request body : 
```
{
  "message" : "Hello, how are you feeling?"
}
```

sample response : 
```
{
  "Message": "Hey feebi! I'm feeling alright, just a bit tired. How about you?",
  "Status": "success"
}
```



POST /ingest?index={indexName}

query string parameter : indexName to upload the file into, note that the pinecone index must already be created through the pinecone website prior to the uploading
file form : file name : "file", 

sample response : 
```
File successfully uploaded.
```







