from functools import partial
import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import urllib.request
import tiktoken
from PretrainingData import textToTokenIds, tokenIdsToText, trainModelSimple
from TrainedData import ContinualLearnerWithReplay, formatInput, load_weights_into_gpt
from Transformer import GPTModel, generateText
from downloadModel import download_and_load_gpt2
from huggingface_hub import hf_hub_download


def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def load_apps_dataset(apps_data_path="apps_dataset"):
    """
    Load and process APPS dataset for instruction fine-tuning
    """
    apps_data = []
    
    if not os.path.exists(apps_data_path):
        print(f"APPS dataset not found at {apps_data_path}")
        return []
    
    # APPS dataset structure: train/test folders with problem directories
    for split in ['train', 'test']:
        split_path = os.path.join(apps_data_path, split)
        if not os.path.exists(split_path):
            continue
            
        for problem_dir in os.listdir(split_path):
            problem_path = os.path.join(split_path, problem_dir)
            if not os.path.isdir(problem_path):
                continue
            
            try:
                # Load problem statement
                problem_file = os.path.join(problem_path, 'question.txt')
                if not os.path.exists(problem_file):
                    continue
                    
                with open(problem_file, 'r', encoding='utf-8') as f:
                    problem_statement = f.read().strip()
                
                # Load input/output examples
                input_output_file = os.path.join(problem_path, 'input_output.json')
                if os.path.exists(input_output_file):
                    with open(input_output_file, 'r', encoding='utf-8') as f:
                        io_data = json.load(f)
                    
                    # Format examples
                    examples = ""
                    if 'inputs' in io_data and 'outputs' in io_data:
                        for i, (inp, out) in enumerate(zip(io_data['inputs'][:3], io_data['outputs'][:3])):
                            examples += f"\nExample {i+1}:\nInput:\n{inp.strip()}\nOutput:\n{out.strip()}\n"
                
                # Load solutions
                solutions_dir = os.path.join(problem_path, 'solutions')
                if os.path.exists(solutions_dir):
                    solution_files = [f for f in os.listdir(solutions_dir) if f.endswith('.py')]
                    if solution_files:
                        # Take the first Python solution
                        with open(os.path.join(solutions_dir, solution_files[0]), 'r', encoding='utf-8') as f:
                            solution_code = f.read().strip()
                        
                        # Format as instruction-following example
                        instruction = f"Solve this competitive programming problem:\n\n{problem_statement}"
                        if examples:
                            instruction += f"\n{examples}"
                        
                        apps_entry = {
                            'instruction': instruction,
                            'input': '',
                            'output': solution_code
                        }
                        apps_data.append(apps_entry)
                        
            except Exception as e:
                print(f"Error processing {problem_path}: {e}")
                continue
    
    return apps_data

def create_apps_entry_from_example(problem_description, input_output_data, solutions):
    """
    Create an instruction entry from APPS-style data
    """
    # Format the problem description with examples
    instruction = f"Solve this competitive programming problem:\n\n{problem_description}"
    
    # Add input/output examples
    if input_output_data.get('inputs') and input_output_data.get('outputs'):
        instruction += "\n\nExamples:"
        for i, (inp, out) in enumerate(zip(input_output_data['inputs'][:3], input_output_data['outputs'][:3])):
            instruction += f"\n\nExample {i+1}:\nInput:\n{inp.strip()}\nOutput:\n{out.strip()}"
    
    # Use the first solution as the target output
    solution_code = solutions[0] if solutions else ""
    
    return {
        'instruction': instruction,
        'input': '',
        'output': solution_code
    }


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data

        self.encoded_texts = []
        for entry in data:
            instructionPlusInput = formatInput(entry)
            responseText = f"\n\n### Response:\n{entry['output']}"
            fullText = instructionPlusInput + responseText
            self.encoded_texts.append(
                tokenizer.encode(fullText)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


def customCollate(batch, padTokenId = 50256, ignoreIndex =-100, allowedMaxLength = None, device = "cpu"):
    batchMaxLength = max(len(item) + 1 for item in batch)

    inputsList, targetsList = [], []

    for item in batch:
        newItem = item.copy()
        newItem += [padTokenId]

        padded = (
            newItem + [padTokenId] * (batchMaxLength - len(newItem))
        )
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == padTokenId
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignoreIndex

        if allowedMaxLength is not None:
            inputs = inputs[:allowedMaxLength]
            targets = targets[:allowedMaxLength]

        inputsList.append(inputs)
        targetsList.append(targets)
    inputsTensor = torch.stack(inputsList).to(device)
    targetsTensor = torch.stack(targetsList).to(device)
    return inputsTensor, targetsTensor



def main():

    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    
    data = download_and_load_file(file_path, url)
    
    print("Number of entries:", len(data))

    try:
        filePath = hf_hub_download(
            repo_id="kaist-ai/CoT-Collection",
            filename="data/CoT_collection_en.json",
            repo_type="dataset"
        )

        # Load and inspect the JSON structure
        with open(filePath, 'r', encoding='utf-8') as f:
            data2 = json.load(f)

        print("Data type:", type(data2))
        print("Data structure:")
        if isinstance(data2, list) and len(data2) > 0:
            # Process the CoT data to match the expected format
            processed_data = []
            for item in data2:
                if isinstance(item, dict):
                    # Convert to instruction format if possible
                    processed_item = {
                        'instruction': item.get('instruction', ''),
                        'input': item.get('input', ''),
                        'output': item.get('output', '')
                    }
                    # Only add if it has the required fields
                    if processed_item['instruction'] and processed_item['output']:
                        processed_data.append(processed_item)
            
            print(f"Processed {len(processed_data)} items from CoT dataset")
            data.extend(processed_data)  # Use extend instead of +=
    except Exception as e:
        print(f"Error loading CoT dataset: {e}")
        print("Continuing with original data only...")

      # Load APPS dataset
    print("\n=== Loading APPS Dataset ===")
    apps_data = load_apps_dataset("apps_dataset")  # Adjust path as needed
    
    if apps_data:
        print(f"Loaded {len(apps_data)} APPS problems")
        data.extend(apps_data)
    else:
        print("No APPS data found, creating sample from provided example...")
        
        # Create sample APPS entry from your provided example
        problem_description = """Mikhail walks on a Cartesian plane. He starts at the point $(0, 0)$, and in one move he can go to any of eight adjacent points. For example, if Mikhail is currently at the point $(0, 0)$, he can go to any of the following points in one move:   $(1, 0)$;  $(1, 1)$;  $(0, 1)$;  $(-1, 1)$;  $(-1, 0)$;  $(-1, -1)$;  $(0, -1)$;  $(1, -1)$. 

        If Mikhail goes from the point $(x1, y1)$ to the point $(x2, y2)$ in one move, and $x1 \\ne x2$ and $y1 \\ne y2$, then such a move is called a diagonal move.

        Mikhail has $q$ queries. For the $i$-th query Mikhail's target is to go to the point $(n_i, m_i)$ from the point $(0, 0)$ in exactly $k_i$ moves. Among all possible movements he want to choose one with the maximum number of diagonal moves. Your task is to find the maximum number of diagonal moves or find that it is impossible to go from the point $(0, 0)$ to the point $(n_i, m_i)$ in $k_i$ moves.

        Note that Mikhail can visit any point any number of times (even the destination point!).

        -----Input-----
        The first line of the input contains one integer $q$ ($1 \\le q \\le 10^4$) — the number of queries.
        Then $q$ lines follow. The $i$-th of these $q$ lines contains three integers $n_i$, $m_i$ and $k_i$ ($1 \\le n_i, m_i, k_i \\le 10^{18}$) — $x$-coordinate of the destination point of the query, $y$-coordinate of the destination point of the query and the number of moves in the query, correspondingly.

        -----Output-----
        Print $q$ integers. The $i$-th integer should be equal to -1 if Mikhail cannot go from the point $(0, 0)$ to the point $(n_i, m_i)$ in exactly $k_i$ moves described above. Otherwise the $i$-th integer should be equal to the the maximum number of diagonal moves among all possible movements."""

        sample_input_output = {
            "inputs": ["3\n2 2 3\n4 3 7\n10 1 9\n"],
            "outputs": ["1\n6\n-1\n"]
        }
        
        # Use one of your provided solutions
        sample_solution = """q = int(input())
            for i in range(q):
                n, m, k = list(map(int, input().split()))
                if max(n, m) > k:
                    print(-1)
                else:
                    if (n + m) % 2 == 0:
                        if max(n, m) % 2 != k % 2:
                            print(k - 2)
                        else:
                            print(k)
                    else:
                        print(k - 1)"""
        
        sample_apps_entry = create_apps_entry_from_example(
            problem_description, 
            sample_input_output, 
            [sample_solution]
        )
        
        data.append(sample_apps_entry)
        print("Added sample APPS problem")
    
    
    print("Number of entries:", len(data))
    print(formatInput(data[999]))
    
    modelInput = formatInput(data[999])
    desiredResponse = f"\n\n### Response:\n{data[999]['output']}"
    print(modelInput + desiredResponse)

    trainPortion = int(len(data) * 0.85)
    testPortion = int(len(data) * 0.1)
    valPortion = len(data) - trainPortion - testPortion

    trainData = data[:trainPortion]
    testData = data[trainPortion:trainPortion + testPortion]
    valData = data[trainPortion + testPortion:]

    print("Training set length:", len(trainData))
    print("Validation set length:", len(valData))
    print("Test set length:", len(testData))

    tokenizer = tiktoken.get_encoding("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    customizedCollate = partial(customCollate, device = device, allowedMaxLength = 1024)

    numWorkers = 0 
    batchSize = 8

    trainDataset = InstructionDataset(trainData, tokenizer)
    trainLoader = DataLoader(trainDataset, batch_size= batchSize, collate_fn=customizedCollate, shuffle=True, drop_last=True, num_workers=numWorkers)

    valDataset = InstructionDataset(valData, tokenizer)
    valLoader = DataLoader(valDataset, batch_size= batchSize, collate_fn=customizedCollate, shuffle=False, drop_last=False, num_workers=numWorkers)

    testDataset = InstructionDataset(testData, tokenizer)
    testLoader = DataLoader(testDataset, batch_size= batchSize, collate_fn=customizedCollate, shuffle=False, drop_last=False, num_workers=numWorkers)
    
    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG = {
        "vocabSize": 50257,     # Vocabulary size
        "contextLength": 1024,  # Context length
        "embDim": 768,
        "nHeads": 12,
        "nLayers": 12,
        "dropRate": 0.1,        # Dropout rate
        "qkvBias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"embDim": 768, "nLayers": 12, "nHeads": 12},
        "gpt2-medium (355M)": {"embDim": 1024, "nLayers": 24, "nHeads": 16},
        "gpt2-large (774M)": {"embDim": 1280, "nLayers": 36, "nHeads": 20},
        "gpt2-xl (1558M)": {"embDim": 1600, "nLayers": 48, "nHeads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    modelSize = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    
    setting, params = download_and_load_gpt2(
        model_size=modelSize,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create continual learner with replay
    learner = ContinualLearnerWithReplay(model, tokenizer, BASE_CONFIG, device)
    
    # Try to get pretrained parameters
    pretrained_params = None
    try:
        modelSize = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        setting, pretrained_params = download_and_load_gpt2(model_size=modelSize, models_dir="gpt2")
        print(f"✓ Pretrained {modelSize} parameters loaded")
    except Exception as e:
        print(f"⚠ Could not load pretrained parameters: {e}")
    
    # Smart weight initialization
    print("\n=== Initializing Model Weights ===")
    load_method = learner.initialize_weights(
        pretrained_params=pretrained_params,
        checkpoint_path="continual_model_latest.pth"
    )
    
    # Load previous state if available
    if os.path.exists("latest_model_checkpoint.pth"):
        learner.load_state(
            "latest_model_checkpoint.pth", 
            "latest_model_replay_buffer.json"
        )

    def process_new_data_with_replay(new_data):
        """Process new data with experience replay"""
        losses = learner.add_new_data_and_train_with_replay(
            new_data,
            num_epochs=1,
            learning_rate=0.00001,  # Lower learning rate
            batch_size=6
        )
        
        # Save state
        checkpoint_file, buffer_file = learner.save_state("continual_model_replay")
        
        # Update latest files
        for old_file, new_file in [
            ("latest_model_checkpoint.pth", checkpoint_file),
            ("latest_model_replay_buffer.json", buffer_file)
        ]:
            if os.path.exists(old_file):
                os.remove(old_file)
            os.rename(new_file, old_file)
        
        return losses
    
    losses = process_new_data_with_replay(trainData)

    non_apps_data = []
    apps_training_data = []
    
    for item in trainData:
        if 'competitive programming' in item.get('instruction', '').lower():
            apps_training_data.append(item)
        else:
            non_apps_data.append(item)
    
    print(f"First training on {len(non_apps_data)} non-APPS examples")
    if non_apps_data:
        losses1 = process_new_data_with_replay(non_apps_data)
    
    print(f"Then incrementally adding {len(apps_training_data)} APPS examples")
    if apps_training_data:
        losses2 = process_new_data_with_replay(apps_training_data)

    inputText = formatInput(valData[0])

    tokenIds = generateText(
        model=model, 
        idx=textToTokenIds(inputText, tokenizer),
        maxNewTokens=35,
        contextSize=BASE_CONFIG["contextLength"],
        eosId=50256
    )

    generatedText = tokenIdsToText(tokenIds, tokenizer)

    responseText = (
        generatedText[len(inputText):].replace("### Response:", "").strip()
    )
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    numEpochs = 2

    trainLosses, valLosses, tokensSeen = trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs=numEpochs, evalFreq=5, evalIter=5, startContext=formatInput(valData[0]), tokenizer=tokenizer)

    test_subset = testData[:5]

    for i, entry in tqdm(enumerate(test_subset), total= len(test_subset)):
        inputText = formatInput(entry)

        tokenIds = generateText(
            model=model,
            idx=textToTokenIds(inputText, tokenizer).to(device),
            maxNewTokens=256,
            contextSize=BASE_CONFIG["contextLength"],
            eosId=50256
        )

        generatedText = tokenIdsToText(tokenIds, tokenizer)
        responseText = (
            generatedText[len(inputText):].replace("### Response:", "").strip()
        )

        test_subset[i]["modelResponse"] = responseText



        print(inputText)
        print(f"\nCorrect response:\n >> {entry['output']}")
        print(f"\nModel response:\n>> {responseText.strip()}")
        print("---------------------------")

    with open("instruction-data-with-response.js", "w") as file:
        json.dump(test_subset, file, indent=4)

    fileName = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), fileName)
    print("Saved model")
    
if __name__ == "__main__":
    main()