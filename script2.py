from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import MongoDBChatMessageHistory
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

from helpers import printMessage

from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()

mongo_connection_string = "mongodb://root:example@localhost:27017"

# llm = OpenAI(verbose=True)
llm = ChatOpenAI(verbose=True)

template = """
You are a chatbot having a conversation with a human.
{history}
Human: {input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
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

for message in message_history.messages[-3:]:
    printMessage(message)

# original: e6c9e418-3e0c-47b3-bcea-962a9b37e6a5
# nova: fe5466fa-176e-4663-aabb-ab23852b2326

# Memoria sem controle de tamanho, uma hora estoura limite de tokens
# memory = ConversationBufferMemory(chat_memory=message_history, memory_key="history")
# chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

# Memória baseada em histórico resumido. IA perde detalhes da conversa quando a mesma é recarregada
# memory = ConversationSummaryMemory(chat_memory=message_history, memory_key="history", llm=llm)
# memory.buffer = memory.predict_new_summary(messages=message_history.messages, existing_summary="")


memory = ConversationSummaryBufferMemory(chat_memory=message_history, memory_key="history", llm=llm)

chain = ConversationChain(llm=llm, memory=memory, verbose=True)

while True:
    # Capture user input
    user_input = input("You: ")

    response = chain.predict(input=user_input)

    print(f"Chatbot: {response}")
