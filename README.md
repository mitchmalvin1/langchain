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

Run `aws configure` and fill in the AWS Access Key ID, AWS Secret Access Key and Region.

Do not delete the empty directories `/historyData` and `/ingestData` in your folder as the `ingest` and `save` APIs will utilize these dummy directories.


Navigate to the  directory and simply run :
```
flask run
```

You can comment out certain parts of the `print` statement in the code and remove the `verbose=True` property in the `ConversationalRetrievalChain` if you do not wish to see so much logging that is cluttering your CLI.

The `ingestion-template.txt` in the root directory is the sample txt file that one can specify when calling the `ingest/template` API. Do not that each key (i.e user,influencer,entities etc) should be in its own line.

**API foot print** : 

**POST** `/startChat?user={user}&influencer={influencer}`
This API will initialize a conversation chain between a user and an influencer. This API must be triggered before any `chat` API call.
`localhost:5000/startchat?user=mitch&influencer=fabian`

**POST** `/chat?user={user}&influencer={influencer}` 
This API will send a message to chat with the bot after the conversation has been initialized.
`localhost:5000/chat?user=mitch&influencer=fabian`
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

**POST** `/save?user={user}&influencer={influencer}` 
This API will save all the chat history between the influencer and the user so far and upload it onto S3 buckets.
`localhost:5000/save?user=mitch&influencer=fabian`



**POST** `/ingest?index={indexName}&influencer={influencer}`
This API will take in a file form and upload the file onto Pinecone index and namespace of {influencer}

file form : 
-file name : "file", 

sample response : 
```
File successfully uploaded.
```

**POST** `ingest/template?influencer={influencer}`
This API will take in a file form of ingestion template and upload the file onto S3 bucket named `digital-immortality-ingestion-template` and path of `/{influencer}`

file form : 
-file name : "file", 

sample response : 
```
File successfully uploaded.
```







