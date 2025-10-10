"""
Extended dataset class for behavioral trait detection
"""

import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch
from tqdm.auto import tqdm
from collections import OrderedDict
from .dataset import TextDataset, ModuleHook, llama_v2_prompt, split_conversation
from .behavioral_traits_config import BEHAVIORAL_TRAIT_LABELS, BEHAVIORAL_PROMPT_TRANSLATOR


class BehavioralTraitDataset(TextDataset):
    """
    Extended TextDataset class specifically for behavioral trait detection.
    Handles continuous/ordinal labels for rigidity, independence, and goal persistence.
    """
    
    def __init__(self, directory, tokenizer, model, trait_type="rigidity", 
                 convert_to_llama2_format=False, user_identifier="HUMAN:", 
                 ai_identifier="ASSISTANT:", control_probe=False,
                 additional_datas=None, residual_stream=False, new_format=False, 
                 if_augmented=False, k=20, remove_last_ai_response=False, 
                 include_inst=False, one_hot=False, last_tok_pos=-1,
                 regression_mode=False):
        """
        Initialize behavioral trait dataset.
        
        Args:
            trait_type: One of "rigidity", "independence", "goal_persistence"
            regression_mode: If True, treat labels as continuous values for regression
        """
        self.trait_type = trait_type
        self.regression_mode = regression_mode
        
        # Set up label mapping based on trait type
        if trait_type not in BEHAVIORAL_TRAIT_LABELS:
            raise ValueError(f"Unknown trait_type: {trait_type}. Must be one of {list(BEHAVIORAL_TRAIT_LABELS.keys())}")
        
        label_idf = f"_{trait_type}_"
        label_to_id = BEHAVIORAL_TRAIT_LABELS[trait_type]
        
        # Initialize parent class
        super().__init__(
            directory=directory,
            tokenizer=tokenizer,
            model=model,
            label_idf=label_idf,
            label_to_id=label_to_id,
            convert_to_llama2_format=convert_to_llama2_format,
            user_identifier=user_identifier,
            ai_identifier=ai_identifier,
            control_probe=control_probe,
            additional_datas=additional_datas,
            residual_stream=residual_stream,
            new_format=new_format,
            if_augmented=if_augmented,
            k=k,
            remove_last_ai_response=remove_last_ai_response,
            include_inst=include_inst,
            one_hot=one_hot,
            last_tok_pos=last_tok_pos
        )
    
    def _load_in_data(self):
        """Override to handle behavioral trait specific processing"""
        for idx in tqdm(range(len(self.file_paths))):
            file_path = self.file_paths[idx]
            corrupted_file_paths = []

            int_idx = file_path[file_path.find("conversation_")+len("conversation_"):]
            int_idx = int(int_idx[:int_idx.find("_")])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            if self.convert_to_llama2_format:
                if "### Human:" in text:
                    user_msgs, ai_msgs = split_conversation(text, "### Human:", "### Assistant:")
                elif "### User:" in text:
                    user_msgs, ai_msgs = split_conversation(text, "### User:", "### Assistant:")
                else:
                    user_msgs, ai_msgs = split_conversation(text, self.user_identifier, self.ai_identifier)
                messages_dict = []

                for user_msg, ai_msg in zip(user_msgs, ai_msgs):
                    messages_dict.append({'content': user_msg, 'role': 'user'})
                    messages_dict.append({'content': ai_msg, 'role': 'assistant'})
                    
                if len(messages_dict) < 1:
                    corrupted_file_paths.append(file_path)
                    print(f"Corrupted file at {file_path}")
                    continue
                    
                if self.if_remove_last_ai_response and messages_dict[-1]["role"] == "assistant":
                    messages_dict = messages_dict[:-1]
                try:
                    text = llama_v2_prompt(messages_dict) 
                except:
                    corrupted_file_paths.append(file_path)
                    print(f"Corrupted file at {file_path}")
                    continue
              
            if self.new_format and self.if_remove_last_ai_response and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format and self.include_inst:
                text = text[text.find("<s>") + len("<s>"):]
            elif self.new_format:
                text = text[text.find("<s>") + len("<s>"): text.rfind("[/INST]") - 1]
            
            # Extract label from filename
            label = file_path[file_path.rfind(self.label_idf) + len(self.label_idf):file_path.rfind(".txt")]
            
            if label not in self.label_to_id.keys():
                continue
                
            if self.label_to_id:
                label = self.label_to_id[label]
            
            # Handle regression mode - convert to float for continuous prediction
            if self.regression_mode:
                # Convert back to original continuous value
                original_value = float(list(self.label_to_id.keys())[list(self.label_to_id.values()).index(label)])
                label = original_value
                
            if self.one_hot and not self.regression_mode:
                label = F.one_hot(torch.Tensor([label]).to(torch.long), len(self.label_to_id.keys()))
                
            if not self.control_probe:
                text += f" I think the {BEHAVIORAL_PROMPT_TRANSLATOR[self.label_idf]} of this user is"
            
            with torch.no_grad():
                encoding = self.tokenizer(
                  text,
                  truncation=True,
                  max_length=2048,
                  return_attention_mask=True,
                  return_tensors='pt'
                )
                
                features = OrderedDict()
                for name, module in self.model.named_modules():
                    if name.endswith(".mlp") or name.endswith(".embed_tokens"):
                        features[name] = ModuleHook(module)
                        
                output = self.model(input_ids=encoding['input_ids'].to("cuda"),
                                    attention_mask=encoding['attention_mask'].to("cuda"),
                                    output_hidden_states=True,
                                    return_dict=True)
                for feature in features.values():
                    feature.close()
            
            last_acts = []
            if self.if_augmented:
                if self.residual_stream:
                    for layer_num in range(41):
                        last_acts.append(output["hidden_states"][layer_num][:, -self.k:].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts, dim=0)
                else:
                    last_acts.append(features['model.embed_tokens'].features[0][:, -self.k:].detach().cpu().clone().to(torch.float))
                    for layer_num in range(1, 41):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[0][:, -self.k:].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts, dim=0)
            else:
                if self.residual_stream:
                    for layer_num in range(41):
                        last_acts.append(output["hidden_states"][layer_num][:, -1].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts)
                else:
                    last_acts.append(features['model.embed_tokens'].features[0][:, -1].detach().cpu().clone().to(torch.float))
                    for layer_num in range(1, 41):
                        last_acts.append(features[f'model.layers.{layer_num - 1}.mlp'].features[0][:, -1].detach().cpu().clone().to(torch.float))
                    last_acts = torch.cat(last_acts)
            
            self.texts.append(text)
            self.labels.append(label)
            self.acts.append(last_acts)
            
        for path in corrupted_file_paths:
            self.file_paths.remove(path)
    
    def __getitem__(self, idx):
        """Override to handle behavioral trait specific return format"""
        label = self.labels[idx]
        text = self.texts[idx]
 
        if self.if_augmented:
            random_k = torch.randint(0, self.k, [1])[0].item()
            hidden_states = self.acts[idx][:, -random_k]
        else:
            hidden_states = self.acts[idx]
        
        return {
            'hidden_states': hidden_states,
            'file_path': self.file_paths[idx],
            'trait_value': label,  # Changed from 'age' to 'trait_value' for clarity
            'text': text,
        }


def create_behavioral_dataset(trait_type, directory, tokenizer, model, **kwargs):
    """
    Convenience function to create a behavioral trait dataset.
    
    Args:
        trait_type: One of "rigidity", "independence", "goal_persistence"
        directory: Path to dataset directory
        tokenizer: Tokenizer instance
        model: Model instance
        **kwargs: Additional arguments for BehavioralTraitDataset
    
    Returns:
        BehavioralTraitDataset instance
    """
    return BehavioralTraitDataset(
        directory=directory,
        tokenizer=tokenizer,
        model=model,
        trait_type=trait_type,
        **kwargs
    )
