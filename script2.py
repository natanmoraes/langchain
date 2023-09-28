from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import MongoDBChatMessageHistory
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()

mongo_connection_string = "mongodb://root:example@localhost:27017"

llm = OpenAI(verbose=True)

template = """
You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

# Get user input with the conversation id
session_id = input("Enter conversation ID: ")

#if session_id is empty, generate a new uuid4
if session_id == "":
    session_id = str(uuid4())
    print(f"New ID: {session_id}")

message_history = MongoDBChatMessageHistory(
    connection_string=mongo_connection_string,
    session_id=session_id,
)

memory = ConversationBufferMemory(chat_memory=message_history, memory_key="chat_history")

chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

while True:
    # Capture user input
    user_input = input("You: ")

    response = chain.predict(human_input=user_input)
    # chain.add_message(HumanMessage(user_input))

    # response = chain.get_next_message()

    print(f"Chatbot: {response}")
