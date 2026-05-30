import requests
import json
import sys

BASE_URL = "http://localhost:11435"

def check_model():
    r = requests.get(f"{BASE_URL}/api/tags")
    models = r.json().get("models", [])
    if not models:
        print("No model loaded in opengllama. Load one first.")
        sys.exit(1)
    return models[0]["name"]

def chat(messages):
    print("Assistant: ", end="", flush=True)
    r = requests.post(f"{BASE_URL}/api/chat",
                      json={"messages": messages, "stream": True},
                      stream=True)
    full = ""
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("done"):
            break
        piece = data.get("message", {}).get("content", "")
        full += piece
        print(piece, end="", flush=True)
    print("\n")
    return full

def main():
    model = check_model()
    print(f"Connected to: {model}")
    print("Type your messages. /clear to reset, Ctrl+C to quit.\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input == "/clear":
            history.clear()
            print("Chat history cleared.\n")
            continue
        history.append({"role": "user", "content": user_input})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
