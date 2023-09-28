from langchain.schema.messages import (BaseMessage, AIMessage)

def printMessage(message: BaseMessage):
  if isinstance(message, AIMessage):
    prefix = "AI: "
  else:
    prefix = "You: "
  print(prefix+message.content)
