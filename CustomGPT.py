from llama_cpp import Llama

my_model_path = "C:/Users/Lindelwe/Desktop/CustomGPT/model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
CONTEXT_SIZE = 512

model = Llama(model_path=my_model_path,n_ctx=CONTEXT_SIZE)

def generateText(userInput,
                 max_tokens=150,
                 temperature=0.7,
                 top_p=0.2,
                 echo=True,
                 stop=["QuitChat()"]
                 ):

    #define the parameters
    modelOutput = model(userInput,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        echo=echo,
                        stop=stop
                        )

    return modelOutput["choices"][0]["text"].strip()

if __name__ == "__main__":
    print("\n......................Welcome to CustomGPT..................")

    while(True):
        prompt = input("\nEnter Message: ")
        print("Generating text...")
        text = generateText(prompt)
        print(f"\n {text}")
