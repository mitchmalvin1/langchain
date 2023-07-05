# langchain

## To initialize

Get the API key from OPEN AI's website and set it in your OS envrionment variable

For instance : 

```
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
echo $OPENAI_API_KEY
```

Make sure you have `pip` installed in your device, it is just a package manager for python, then run

```
pip install langchain
pip install openai
```


Copy all files (you can find in our group) and name them as follows :
```
fabian_chat.txt : the file containing fabian's entire conversation
chat_summary.txt : the file containing the small chunk from fabian's conversation used to generate the profile
```

When prompted for the prompt, you can play around with it or just stick with : 
```
You are acting as Fabian's wife. Analyze the given conversation in the context. I will talk to you as Fabian and you will reply me as Fabian's wife. Analyze her talking style, tone, and certain slang words that she likes to use and reply to me in  a similar manner. You are to reply to me only once and only as Fabian's wife. Do not complete the conversation more than once. If there is not enough information, try to infer from the context and reply to me on your own. Try to imitate the talking style provided below sparingly and only when it is appropriate to do so.
```

Navigate to the correct directory and simply run :
```
python main.py
```

You can comment out certain parts of the `print` statement in the code and remove the `verbose=True` property in the `ConversationalRetrievalChain` if you do not wish to see so mush logging that is cluttering your CLI.
