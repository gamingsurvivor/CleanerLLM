import time
from functools import partial
import json
import os
import torch
import numpy as np
import random
from collections import deque
from torch.utils.data import DataLoader, Dataset

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.posEmb.weight = assign(gpt.posEmb.weight, params['wpe'])
    gpt.tokEmb.weight = assign(gpt.tokEmb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        qkv_w = np.concatenate([q_w, k_w,v_w], axis=-1)
        gpt.trfBlocks[b].att.qkv.weight = assign(
            gpt.trfBlocks[b].att.qkv.weight, qkv_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3,axis=-1
        )

        qkv_b = np.concatenate([q_b, k_b, v_b], axis=-1)
        gpt.trfBlocks[b].att.qkv.bias = assign(
            gpt.trfBlocks[b].att.qkv.bias, qkv_b)

        # Load output projection weights (this stays the same)
        gpt.trfBlocks[b].att.proj.weight = assign(
            gpt.trfBlocks[b].att.proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trfBlocks[b].att.proj.bias = assign(
            gpt.trfBlocks[b].att.proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # Feed-forward layers (unchanged)
        gpt.trfBlocks[b].ff.layers[0].weight = assign(
            gpt.trfBlocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trfBlocks[b].ff.layers[0].bias = assign(
            gpt.trfBlocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trfBlocks[b].ff.layers[2].weight = assign(
            gpt.trfBlocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trfBlocks[b].ff.layers[2].bias = assign(
            gpt.trfBlocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # Layer norm parameters (unchanged)
        gpt.trfBlocks[b].norm1.scale = assign(
            gpt.trfBlocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trfBlocks[b].norm1.shift = assign(
            gpt.trfBlocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trfBlocks[b].norm2.scale = assign(
            gpt.trfBlocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trfBlocks[b].norm2.shift = assign(
            gpt.trfBlocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    # Final layer norm and output head (unchanged)
    gpt.finalNorm.scale = assign(gpt.finalNorm.scale, params["g"])
    gpt.finalNorm.shift = assign(gpt.finalNorm.shift, params["b"])
    gpt.outHead.weight = assign(gpt.outHead.weight, params["wte"])


class SmartWeightLoader:
    """Enhanced weight loading system that wraps your existing function"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def load_pretrained_gpt2_weights(self, model, params):
        """Load pretrained GPT-2 weights using your original function"""
        print("Loading pretrained GPT-2 weights...")
        load_weights_into_gpt(model, params)
        model.to(self.device)
        print("✓ Pretrained GPT-2 weights loaded successfully")
        return "pretrained_loaded"
    
    def load_checkpoint_weights(self, model, checkpoint_path, strict=True):
        """Load weights from a continual learning checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Our continual learning checkpoint format
            state_dict = checkpoint['model_state_dict']
            training_history = checkpoint.get('training_history', [])
            timestamp = checkpoint.get('timestamp', 'unknown')
            
            # Load the state dict
            if strict:
                model.load_state_dict(state_dict)
            else:
                # Load only matching keys, ignore mismatches
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                
                # Report what was loaded
                loaded_keys = set(filtered_dict.keys())
                expected_keys = set(model_dict.keys())
                print(f"  Loaded {len(loaded_keys)}/{len(expected_keys)} parameters")
                if loaded_keys != expected_keys:
                    missing = expected_keys - loaded_keys
                    print(f"  Missing parameters: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            
            model.to(self.device)
            print(f"✓ Checkpoint loaded (training sessions: {len(training_history)})")
            return checkpoint
            
        elif isinstance(checkpoint, dict) and any(k.startswith('trfBlocks') for k in checkpoint.keys()):
            # Simple state dict format
            if strict:
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.to(self.device)
            print("✓ State dict loaded successfully")
            return {"model_state_dict": checkpoint}
        else:
            raise ValueError("Unrecognized checkpoint format")
    
    def save_checkpoint(self, model, training_history=None, config=None, 
                       filename_prefix="continual_model", include_optimizer=False, optimizer=None):
        """Save a complete checkpoint"""
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'training_history': training_history or [],
            'timestamp': timestamp,
            'config': config,
            'pytorch_version': torch.__version__,
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        filename = f"{filename_prefix}_{timestamp}.pth"
        torch.save(checkpoint, filename)
        print(f"✓ Checkpoint saved: {filename}")
        return filename
    
    def smart_load(self, model, pretrained_params=None, checkpoint_path=None, 
                   prefer_checkpoint=True, fallback_to_pretrained=True):
        """Intelligently load weights based on availability"""
        
        # Check what's available
        has_checkpoint = checkpoint_path and os.path.exists(checkpoint_path)
        has_pretrained = pretrained_params is not None
        
        print(f"Weight loading options:")
        print(f"  Checkpoint available: {has_checkpoint}")
        print(f"  Pretrained available: {has_pretrained}")
        print(f"  Prefer checkpoint: {prefer_checkpoint}")
        
        # Try checkpoint first if preferred and available
        if prefer_checkpoint and has_checkpoint:
            try:
                result = self.load_checkpoint_weights(model, checkpoint_path, strict=False)
                return "checkpoint", result
            except Exception as e:
                print(f"⚠ Checkpoint loading failed: {e}")
                if not fallback_to_pretrained:
                    raise
                print("  Falling back to pretrained weights...")
        
        # Try pretrained weights
        if has_pretrained:
            try:
                result = self.load_pretrained_gpt2_weights(model, pretrained_params)
                return "pretrained", {"model_state_dict": model.state_dict()}
            except Exception as e:
                print(f"⚠ Pretrained loading failed: {e}")
                if has_checkpoint:
                    print("  Falling back to checkpoint...")
                    result = self.load_checkpoint_weights(model, checkpoint_path, strict=False)
                    return "checkpoint", result
                else:
                    raise
        
        # Try checkpoint as fallback
        if has_checkpoint:
            try:
                result = self.load_checkpoint_weights(model, checkpoint_path, strict=False)
                return "checkpoint", result
            except Exception as e:
                print(f"⚠ All loading methods failed. Using random initialization.")
                return "random", {"model_state_dict": model.state_dict()}
        
        # If nothing works, keep random initialization
        print("⚠ No weights available. Using random initialization.")
        return "random", {"model_state_dict": model.state_dict()}


class ContinualLearner:
    """Enhanced continual learner with smart weight loading"""
    
    def __init__(self, model, tokenizer, base_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.base_config = base_config
        self.device = device
        self.training_history = []
        self.weight_loader = SmartWeightLoader(device)
        
    def initialize_weights(self, pretrained_params=None, checkpoint_path="continual_model_latest.pth"):
        """Initialize model weights using smart loading"""
        
        load_method, result = self.weight_loader.smart_load(
            self.model,
            pretrained_params=pretrained_params,
            checkpoint_path=checkpoint_path,
            prefer_checkpoint=True,
            fallback_to_pretrained=True
        )
        
        # Extract training history if available
        if isinstance(result, dict) and 'training_history' in result:
            self.training_history = result['training_history']
            print(f"  Restored training history: {len(self.training_history)} sessions")
        
        return load_method
    
    def save_state(self, filename_prefix="continual_model", include_optimizer=False, optimizer=None):
        """Save model state"""
        return self.weight_loader.save_checkpoint(
            self.model,
            training_history=self.training_history,
            config=self.base_config,
            filename_prefix=filename_prefix,
            include_optimizer=include_optimizer,
            optimizer=optimizer
        )
    
    def add_training_session_record(self, session_info):
        """Add a training session to history"""
        session_info['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.training_history.append(session_info)


class ExperienceReplayBuffer:
    def __init__(self, max_size=10000, replay_ratio=0.3):
        """
        max_size: Maximum number of examples to store
        replay_ratio: Ratio of old examples to mix with new data
        """
        self.buffer = deque(maxlen=max_size)
        self.replay_ratio = replay_ratio
        
    def add_examples(self, new_examples):
        """Add new examples to the buffer"""
        self.buffer.extend(new_examples)
        print(f"Added {len(new_examples)} examples. Buffer size: {len(self.buffer)}")
        
    def get_training_batch(self, new_examples):
        """Get a training batch mixing new and old examples"""
        if len(self.buffer) == 0:
            return new_examples
            
        # Calculate how many old examples to include
        num_replay = int(len(new_examples) * self.replay_ratio)
        
        if num_replay > len(self.buffer):
            num_replay = len(self.buffer)
            
        # Sample random examples from buffer
        replay_examples = random.sample(list(self.buffer), num_replay)
        
        # Combine new and old examples
        combined_batch = new_examples + replay_examples
        random.shuffle(combined_batch)  # Shuffle the combined batch
        
        print(f"Training batch: {len(new_examples)} new + {len(replay_examples)} replay = {len(combined_batch)} total")
        
        return combined_batch
        
    def save_buffer(self, filepath):
        """Save buffer to disk"""
        with open(filepath, 'w') as f:
            json.dump(list(self.buffer), f, indent=2)
            
    def load_buffer(self, filepath):
        """Load buffer from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.buffer.extend(data)
                print(f"Loaded {len(data)} examples into buffer")

class ContinualLearnerWithReplay(ContinualLearner):
    def __init__(self, model, tokenizer, base_config, device, buffer_size=10000):
        super().__init__(model, tokenizer, base_config, device)
        self.replay_buffer = ExperienceReplayBuffer(max_size=buffer_size)
        
    def add_new_data_and_train_with_replay(self, new_data, num_epochs=1, 
                                         learning_rate=0.00005, batch_size=8):
        """Train with experience replay to reduce catastrophic forgetting"""
        
        # Get training batch with replay
        training_batch = self.replay_buffer.get_training_batch(new_data)
        
        # Add new examples to buffer for future replay
        self.replay_buffer.add_examples(new_data)
        
        # Create dataset with mixed data
        mixed_dataset = InstructionDataset(training_batch, self.tokenizer)
        
        customized_collate = partial(customCollate, device=self.device, allowedMaxLength=1024)
        
        mixed_loader = DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate,
            shuffle=True,
            drop_last=True
        )
        
        # Use slightly higher learning rate since we're mixing old and new data
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate * 0.5,  # Medium learning rate
            weight_decay=0.1
        )
        
        print(f"Training with replay on {len(training_batch)} total examples...")
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(mixed_loader):
                optimizer.zero_grad()
                
                logits = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), ignore_index=-100
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Record training session
        training_session = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'num_new_samples': len(new_data),
            'num_total_samples': len(training_batch),
            'buffer_size': len(self.replay_buffer.buffer),
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'final_loss': epoch_losses[-1] if epoch_losses else None
        }
        
        self.training_history.append(training_session)
        self.model.eval()
        
        return epoch_losses
    
    def save_state(self, filename_prefix="continual_model_with_replay", include_optimizer=False, optimizer=None):
        """Save both model and replay buffer"""
        # Save model checkpoint using parent class method
        checkpoint_file = super().save_state(
            filename_prefix=filename_prefix,
            include_optimizer=include_optimizer,
            optimizer=optimizer
        )
        
        # Save replay buffer
        buffer_file = f"{filename_prefix}_replay_buffer_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.json"
        self.replay_buffer.save_buffer(buffer_file)
        
        print(f"✓ Replay buffer saved: {buffer_file}")
        
        return checkpoint_file, buffer_file
    
    def load_state(self, checkpoint_path, buffer_path=None):
        """Load both model and replay buffer"""
        # Load model weights using the weight loader
        try:
            result = self.weight_loader.load_checkpoint_weights(
                self.model, 
                checkpoint_path, 
                strict=False
            )
            
            # Extract training history if available
            if isinstance(result, dict) and 'training_history' in result:
                self.training_history = result['training_history']
                print(f"  Restored training history: {len(self.training_history)} sessions")
                
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            raise
        
        # Load replay buffer if path provided
        if buffer_path and os.path.exists(buffer_path):
            self.replay_buffer.load_buffer(buffer_path)
            print(f"✓ Replay buffer loaded from: {buffer_path}")
        elif buffer_path:
            print(f"⚠ Replay buffer file not found: {buffer_path}")
    
    def initialize_weights_with_replay(self, pretrained_params=None, checkpoint_path="continual_model_with_replay_latest.pth", buffer_path=None):
        """Initialize model weights and load replay buffer if available"""
        
        # Initialize model weights using parent method
        load_method = super().initialize_weights(pretrained_params, checkpoint_path)
        
        # Try to load replay buffer
        if buffer_path and os.path.exists(buffer_path):
            self.replay_buffer.load_buffer(buffer_path)
        else:
            # Try to find a buffer file based on checkpoint name
            if checkpoint_path:
                base_name = checkpoint_path.replace('.pth', '')
                potential_buffer_path = f"{base_name}_replay_buffer.json"
                if os.path.exists(potential_buffer_path):
                    self.replay_buffer.load_buffer(potential_buffer_path)
                    print(f"✓ Auto-loaded replay buffer: {potential_buffer_path}")
        
        return load_method

def formatInput(entry):
    instructionText = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    inputText = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    return instructionText + inputText


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