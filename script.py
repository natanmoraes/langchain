from langchain.memory import MongoDBChatMessageHistory
from langchain.chat_models import ChatOpenAI
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from helpers import printMessage

from dotenv import load_dotenv
import uuid

load_dotenv()

# chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
chat = ChatOpenAI()

mongo_connection_string = "mongodb://root:example@localhost:27017"
session_id = ''

print("Welcome.")
print("- Type 'new' to start a new conversation.")
print("- Type 'load' to load a conversation by id.")

# Capture user input
user_input = input("What would you like to do? ")
if user_input == "new":
    new_session_id = uuid.uuid4()
    session_id = str(new_session_id)
    is_new_session = True
elif user_input == "load":
    loaded_session_id = input("Enter conversation ID: ")
    session_id = str(loaded_session_id)
    is_new_session = False
else:
    print("Invalid input. Exiting")
    exit()

message_history = MongoDBChatMessageHistory(
    connection_string=mongo_connection_string,
    session_id=session_id,
)

# If new session, display a welcome message
if is_new_session:
    print(f"Starting new conversation with id {session_id}.")
    message = "Welcome to the chat. How can I help you?"
    print(f"AI: {message}")
    message_history.add_ai_message(message)
else:
    # If not new session, load previous messages
    print(f"Loaded conversation {session_id}:")
    for message in message_history.messages:
        printMessage(message)

# Start the conversation loop
while True:
    # Capture user input
    user_input = input("You: ")

    # Add user message to message history
    message_history.add_user_message(user_input)

    # Generate response
    response = chat(message_history.messages)

    # Add AI message to message history
    message_history.add_message(response)

    # Print response
    printMessage(response)
