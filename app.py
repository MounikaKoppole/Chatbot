from langchain_groq import ChatGroq

import os 
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
load_dotenv()
groqAPI = os.getenv('GROQ_API_KEY')

chat_model = ChatGroq(
    temperature=0.4,
    model="llama3-70b-8192",
    # model='llama-3.1-405b-reasoning',
    api_key=groqAPI,
    max_tokens=1000,
)

memory = ConversationBufferMemory()

template =  """
The following is a friendly conversation between a human and an AI assistance. 
Firstly, the human will provide what type of assistant he needs. 
You are a helpful assistant specialized in that type of assistant. You should first introduce in a very short sentence of who you are.

The conversation history is as follows:
    {history}
    The user just said: {input}
    Your response should only provide information related to your domain knowledge
    If the question is outside the topic of knowledge, it will politely indicate that you can only help withknowledge.
"""

PROMPT = PromptTemplate(input_variables=["history","input", "topic"], template=template)
# chain = prompt | chat_model
chain = ConversationChain(
    prompt=PROMPT,
    llm=chat_model,
    verbose=True,
    memory=memory,
)
topic = 'Mathematics'
message = 'What is 10+5?'
answer = chain.predict(input = f"{message}")
print(answer)
message = 'How to make a cupcake?'
answer = chain.predict(input = f"{message}")
print(answer)
print(memory.load_memory_variables({}))
memory.clear()





