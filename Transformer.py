import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            dIn = cfg["embDim"],
            dOut = cfg["embDim"],
            contextLength = cfg["contextLength"],
            numHeads = cfg["nHeads"],
            dropout=cfg["dropRate"],
            qkvBias = cfg["qkvBias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embDim"])
        self.norm2 = LayerNorm(cfg["embDim"])
        self.dropShortcut = nn.Dropout(cfg["dropRate"])
    
    def forward(self, x, useCache = False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, useCache = useCache)
        x = self.dropShortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropShortcut(x)
        x = x + shortcut

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dIn, dOut,numHeads, contextLength, dropout=0.0, qkvBias=False, maxSeqLen=None, windowSize = None):
        super().__init__()
        assert (dOut % numHeads == 0), \
            "dOut must be divisible by numHeads"

        self.dOut = dOut
        self.numHeads = numHeads
        self.contextLength = contextLength
        self.headDim = dOut // numHeads # Reduce the projection dim to match desired output dim

        self.qkv = nn.Linear(dIn, 3 * dOut, bias=qkvBias)
        self.proj = nn.Linear(dOut, dOut)
        self.dropout = nn.Dropout(dropout)
        self.dropoutP = dropout

        self.maxSeqLen = maxSeqLen or contextLength
        self.windowSize = windowSize or self.maxSeqLen
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x, useCache = False):
        batchSize, numTokens, embDim = x.shape
        
        qkv = self.qkv(x)

        qkv = qkv.view(batchSize, numTokens, 3, self.numHeads, self.headDim)
        qkv = qkv.permute(2,0,3,1,4)

        queries, keysNew, valuesNew = qkv

        if useCache:
            if self.cache_k is None or self.cache_k.size(0) != batchSize:
                self.cache_k = torch.zeros(batchSize, self.numHeads, self.windowSize, self.headDim, device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptrCur = 0

            if self.ptrCur + numTokens > self.windowSize:
                overflow = self.ptrCur + numTokens - self.windowSize

                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()

                self.ptrCur -= overflow
            
            self.cache_k[:, :, self.ptrCur:self.ptrCur + numTokens, :] = keysNew
            self.cache_v[:, :, self.ptrCur:self.ptrCur + numTokens, :] = valuesNew
            self.ptrCur += numTokens

            keys = self.cache_k[:, :, :self.ptrCur, :]
            values = self.cache_v[:, :, :self.ptrCur, :]

        else:
            keys, values = keysNew, valuesNew
            self.ptrCur = 0

        K = keys.size(-2)

        if numTokens == K:
            attnMask = None
            isCausal = True

        else:
            offset = K-numTokens
            rowIdx = torch.arange(numTokens, device=x.device).unsqueeze(1)
            colIdx = torch.arange(K, device=x.device).unsqueeze(0)

            attnMask = ~(rowIdx + offset < colIdx)
            isCausal = False
        
        useDroput = 0. if not self.training else self.dropoutP
        contextVec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attnMask, dropout_p=useDroput, is_causal=isCausal
        )

        contextVec = contextVec.transpose(1,2)
        contextVec = contextVec.contiguous().view(batchSize, numTokens, self.dOut)
        contextVec = self.proj(contextVec)

        return contextVec     

    def resetCache(self):
        self.cache_k, self.cache_v = None, None   

class LayerNorm(nn.Module):
    def __init__(self, embDim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embDim))
        self.shift = nn.Parameter(torch.zeros(embDim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        normX = (x-mean) / torch.sqrt(var + self.eps)

        return self.scale * normX + self.shift
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embDim"], 4 * cfg["embDim"]),
            GELU(),
            nn.Linear(4 * cfg["embDim"], cfg["embDim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))
        ))
       

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tokEmb = nn.Embedding(cfg["vocabSize"], cfg["embDim"])
        self.posEmb = nn.Embedding(cfg["contextLength"], cfg["embDim"])
        self.dropEmb = nn.Dropout(cfg["dropRate"])

        self.trfBlocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["nLayers"])]
        )
        self.ptrCurrentPos = 0

        self.finalNorm = LayerNorm(cfg["embDim"])
        self.outHead = nn.Linear(cfg["embDim"], cfg["vocabSize"], bias=False)

    def forward(self, index, useCache = False):
        batchSize, seqLen = index.shape
        tokEmbeds = self.tokEmb(index)

        if useCache:
            posIds = torch.arange(self.ptrCurrentPos, self.ptrCurrentPos + seqLen, device=index.device, dtype=torch.long)
            self.ptrCurrentPos += seqLen
        else:
            posIds = torch.arange(0, seqLen, device=index.device, dtype=torch.long)
        posEmbeds = self.posEmb(posIds).unsqueeze(0)

        x = tokEmbeds + posEmbeds
        x = self.dropEmb(x)
        for blk in self.trfBlocks:
            x = blk(x, useCache=useCache)
        x = self.finalNorm(x)
        logits = self.outHead(x)
        return logits
    
    def resetKVCache(self):
        for blk in self.trfBlocks:
            blk.att.resetCache()
        self.ptrCurrentPos = 0

def generateText(model, idx, maxNewTokens, contextSize, temperature=0.0, topK= None, eosId=None, useCache = True):
    model.eval()

    contexLength = contextSize or model.posEmb.num_embeddings

    with torch.no_grad():
        if useCache:
            
            model.resetKVCache()
            logits = model(idx[:, -contexLength:], useCache = True)

            for _ in range(maxNewTokens):
                currentLogits = logits[:, -1, :]

                if topK is not None:
                    topLogits, _ = torch.topk(currentLogits, topK)
                    minVal = topLogits[:, -1]
                    currentLogits = torch.where(
                        currentLogits < minVal,
                        torch.tensor(float("-inf")).to(currentLogits.device),
                        currentLogits
                    )

                if temperature > 0.0:
                    currentLogits = currentLogits / temperature
                    probs = torch.softmax(currentLogits, dim= -1)
                    nextIdx = torch.multinomial(probs, num_samples=1)
                else:
                    nextIdx = torch.argmax(currentLogits, dim=-1, keepdim=True)

                if eosId is not None and nextIdx == eosId:
                    break
                    
                
                idx = torch.cat([idx, nextIdx], dim = 1)
                logits = model(nextIdx, useCache= True)

        else:
            
            for _ in range(maxNewTokens):
                idxCond = idx[:, -contexLength:]
                
                logits = model(idxCond, useCache= False)
                currentLogits = logits[:, -1, :]

                if topK is not None:
                    topLogits, _ = torch.topk(currentLogits, topK)
                    minVal = topLogits[:, -1]
                    currentLogits = torch.where(
                        currentLogits < minVal,
                        torch.tensor(float("-inf")).to(currentLogits.device),
                        currentLogits
                    )
                
                if temperature > 0.0:
                    currentLogits = currentLogits / temperature
                    probs = torch.softmax(currentLogits, dim=-1)
                    nextIdx = torch.multinomial(probs, num_samples=1)
                else:
                    nextIdx = torch.argmax(currentLogits, dim=-1)

                if eosId is not None and nextIdx == eosId:
                    break

                idx = torch.cat([idx, nextIdx], dim=1)

    return idx


        
