
import os
import getpass
from flask_cors import CORS
from flask import Flask, request, jsonify
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)


# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

model = init_chat_model("llama3-8b-8192", model_provider="groq")

@app.route("/")
def index():
    return "LLM Chatbot Backend is Running"

@app.route("/chat", methods=["POST"])
def chat():
    # Expecting {"history": [{"role": "human", "content": ...}, {"role": "ai", "content": ...}, ...], "message": ...}
    history = request.json.get("history", [])
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Convert history to LangChain message objects
    message_objs = []
    for item in history:
        if item["role"] == "human":
            message_objs.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            message_objs.append(AIMessage(content=item["content"]))
        # Optionally handle "system" messages etc.

    # Add the user's latest message
    message_objs.append(HumanMessage(content=user_input))

    response = model.invoke(message_objs)
    return jsonify({"response": response.content})

if __name__ == "__main__":
    app.run(debug=True, port=5001)