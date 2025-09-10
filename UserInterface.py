import sys
from pathlib import Path
import tiktoken
import torch
import chainlit

from Transformer import GPTModel, generateText
from PretrainingData import textToTokenIds, tokenIdsToText

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getModelAndTokenizer():

    BASE_CONFIG = {
        "vocabSize": 50257,     # Vocabulary size
        "contextLength": 1024,  # Context length
        "embDim": 1024,
        "nHeads": 16,
        "nLayers": 24,
        "dropRate": 0.0,        # Dropout rate
        "qkvBias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    
    modelPath = Path("gpt2-medium355M-sft.pth")
    if not modelPath.exists():
        print("No model path")

        sys.exit()
    
    checkPoint = torch.load(modelPath, weights_only=True)
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(checkPoint)
    model.to(device)

    return tokenizer, model, BASE_CONFIG

def extractResponse(responseText, inputText):
    return responseText[len(inputText):].replace("### Response:", "").strip()

@chainlit.on_message
async def main(message: chainlit.Message):
    
    tokenizer, model, modelConfig = getModelAndTokenizer()

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    tokenIds = generateText(
        model=model,
        idx=textToTokenIds(prompt, tokenizer).to(device),
        maxNewTokens=35,
        contextSize=modelConfig["contextLength"],
        eosId=50256
    )

    text = tokenIdsToText(tokenIds, tokenizer)
    response = extractResponse(text,prompt)

    await chainlit.Message(
        content=f"{response}",
    ).send()