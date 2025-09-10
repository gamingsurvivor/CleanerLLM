import torch
from Transformer import generateText

def textToTokenIds(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encodedTensor = torch.tensor(encoded).unsqueeze(0)
    return encodedTensor

def tokenIdsToText(tokenIds, tokenizer):
    flat = tokenIds.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calcLossBatch(inputBatch, targetBatch, model, device):
    inputBatch, targetBatch = inputBatch.to(device), targetBatch.to(device)
    logits = model(inputBatch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), targetBatch.flatten())
    return loss

def calcLossLoader(dataLoader, model, device, numBatches = None):
    totalLoss = 0
    if len(dataLoader) == 0:
        return float("Nan")
    elif numBatches is None:
        numBatches = len(dataLoader)
    else:
        numBatches = min(numBatches, len(dataLoader))
    
    for i, (inputBatch, targetBatch) in enumerate(dataLoader):
        if i < numBatches:
            loss = calcLossBatch(inputBatch, targetBatch, model, device)
            totalLoss += loss.item()
        else:
            break

    return totalLoss / numBatches

def evaluateModel(model, trainLoader, valLoader, device, evalIter):
    model.eval()
    with torch.no_grad():
        trainLoss = calcLossLoader(trainLoader, model, device, numBatches=evalIter)
        valLoss = calcLossLoader(valLoader, model, device, numBatches=evalIter)
    model.train()
    return trainLoss, valLoss

def generateAndPrintSample(model, tokenizer, device, startContext):
    model.eval()
    contextSize = model.posEmb.weight.shape[0]
    encoded = textToTokenIds(startContext, tokenizer).to(device)
    with torch.no_grad():
        tokenIds = generateText(
            model=model, idx=encoded,
            maxNewTokens=50, contextSize=contextSize
        )
    decodedText = tokenIdsToText(tokenIds, tokenizer)
    print(decodedText.replace("\n", " "))  # Compact print format
    model.train()

def trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs,
                       evalFreq, evalIter, startContext, tokenizer):
    # Initialize lists to track losses and tokens seen
    trainLosses, valLosses, trackTokensSeen = [], [], []
    tokensSeen, globalStep = 0, -1

    # Main training loop
    for epoch in range(numEpochs):
        model.train()  # Set model to training mode
        
        for inputBatch, targetBatch in trainLoader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calcLossBatch(inputBatch, targetBatch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokensSeen += inputBatch.numel()
            globalStep += 1

            # Optional evaluation step
            if globalStep % evalFreq == 0:
                trainLoss, valLoss = evaluateModel(
                    model, trainLoader, valLoader, device, evalIter)
                trainLosses.append(trainLoss)
                valLosses.append(valLoss)
                trackTokensSeen.append(tokensSeen)
                print(f"Ep {epoch+1} (Step {globalStep:06d}): "
                      f"Train loss {trainLoss:.3f}, Val loss {valLoss:.3f}")

        # Print a sample text after each epoch
        generateAndPrintSample(
            model, tokenizer, device, startContext
        )

    return trainLosses, valLosses, trackTokensSeen
