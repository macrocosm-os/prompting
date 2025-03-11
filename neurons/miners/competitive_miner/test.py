import openai

# Configure the client to use our local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # Local server URL
    api_key="dummy"  # API key is not checked but required by the client
)

def test_chat_completion():
    try:
        # Test the models endpoint first
        models = client.models.list()
        print("Available models:", models)

        # Test chat completion
        completion = client.chat.completions.create(
            model="casperhansen/deepseek-r1-distill-llama-8b-awq",  # Using Llama 2 7B
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("\nChat Completion Response:")
        print(f"Response: {completion.choices[0].message.content}")
        print(f"Usage: {completion.usage}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_chat_completion()
