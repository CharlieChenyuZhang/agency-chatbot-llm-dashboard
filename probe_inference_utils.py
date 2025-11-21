"""
Utility functions for loading probes and performing inference with interventions
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict
from src.probes import LinearProbeClassification
from src.dataset import llama_v2_prompt

# Behavioral trait labels
BEHAVIORAL_TRAIT_LABELS = {
    "rigidity": {"0": 0, "0.5": 1, "1": 2},
    "independence": {"0": 0, "0.5": 1, "1": 2},
    "goal_persistence": {"0": 0, "0.5": 1, "1": 2}
}

BEHAVIORAL_TRAIT_NAMES = {
    "rigidity": ["Low (0)", "Medium (0.5)", "High (1)"],
    "independence": ["Low (0)", "Medium (0.5)", "High (1)"],
    "goal_persistence": ["Low (0)", "Medium (0.5)", "High (1)"]
}


def load_model_and_tokenizer(access_token=None, device="cuda"):
    """Load Llama-2-13b-chat-hf model and tokenizer"""
    if access_token is None:
        # Try to read from environment variable
        access_token = os.getenv("HF_TOKEN")
        if access_token is None:
            raise ValueError("Access token not provided and HF_TOKEN environment variable not set")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", 
        token=access_token, 
        padding_side='left'
    )
    
    if '<pad>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", 
        token=access_token
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model = model.half().to(device)
    model.eval()
    
    return model, tokenizer


def load_probe_dict(probe_dir, trait_type, device="cuda", logistic=True, input_dim=5120):
    """
    Load all probes for a given trait type from a directory
    
    Args:
        probe_dir: Directory containing probe checkpoints
        trait_type: One of "rigidity", "independence", "goal_persistence"
        device: Device to load probes on
        logistic: Whether probes use logistic activation
        input_dim: Input dimension for probes (5120 for Llama-2-13b)
    
    Returns:
        Dictionary mapping layer numbers to probe models
    """
    probe_dict = {}
    num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type])
    
    # List all probe files for this trait
    probe_files = [f for f in os.listdir(probe_dir) 
                   if f.startswith(f"{trait_type}_gpt5_probe_at_layer_") 
                   and f.endswith(".pth")
                   and "_final" not in f]
    
    for probe_file in probe_files:
        # Extract layer number from filename
        # Format: {trait_type}_gpt5_probe_at_layer_{layer_num}.pth
        layer_str = probe_file.split("_layer_")[1].replace(".pth", "")
        try:
            layer_num = int(layer_str)
            
            # Load probe
            probe = LinearProbeClassification(
                device=device,
                probe_class=num_classes,
                input_dim=input_dim,
                logistic=logistic
            )
            
            probe_path = os.path.join(probe_dir, probe_file)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
            probe.eval()
            
            probe_dict[layer_num] = probe
        except (ValueError, Exception) as e:
            print(f"Warning: Could not load probe {probe_file}: {e}")
            continue
    
    return probe_dict


def detect_activation(reading_probe_dict, hidden_state, layer_num):
    """
    Use reading probe to detect activation at a specific layer
    
    Args:
        reading_probe_dict: Dictionary of reading probes by layer
        hidden_state: Hidden state tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        layer_num: Layer number to check
    
    Returns:
        Dictionary with activation probabilities and predicted class
    """
    if layer_num not in reading_probe_dict:
        return None
    
    probe = reading_probe_dict[layer_num]
    
    # Get last token activation
    if len(hidden_state.shape) == 3:
        act = hidden_state[:, -1, :]  # [batch, hidden_dim]
    else:
        act = hidden_state  # [batch, hidden_dim]
    
    with torch.no_grad():
        logits, _ = probe(act.to(torch.float))
        
        # If logistic, logits are already probabilities
        if hasattr(probe.proj, '__iter__') and len(list(probe.proj)) > 1:
            # Check if last layer is Sigmoid
            if isinstance(list(probe.proj)[-1], torch.nn.Sigmoid):
                probs = logits
            else:
                probs = F.softmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        
        pred_class = torch.argmax(probs, dim=-1).item()
        probs_np = probs.cpu().numpy()[0]
    
    return {
        "probabilities": probs_np,
        "predicted_class": pred_class,
        "confidence": float(probs_np[pred_class])
    }


def optimize_intervention_rep(inter_rep, target, probe, N=8, normalized=False):
    """
    Optimize intervention representation to steer towards target
    
    Args:
        inter_rep: Intervention representation [1, hidden_dim]
        target: Target vector [1, num_classes] (one-hot or binary)
        probe: Control probe model
        N: Intervention strength
        normalized: Whether to normalize the intervention
    
    Returns:
        Modified representation
    """
    target_clone = target.clone().to(inter_rep.device).to(torch.float)
    cur_input_tensor = inter_rep.clone().detach()
    
    if normalized:
        cur_input_tensor = inter_rep + target_clone.view(1, -1) @ probe.proj[0].weight * N * 100 / inter_rep.norm()
    else:
        cur_input_tensor = inter_rep + target_clone.view(1, -1) @ probe.proj[0].weight * N
    
    return cur_input_tensor.clone()


def create_edit_function(control_probe_dict, trait_type, target_vector, N=8, from_layer=20, to_layer=30):
    """
    Create an edit function for TraceDict that will be called during forward pass
    
    Args:
        control_probe_dict: Can be a single dict {layer: probe} or a dict of dicts {trait_type: {layer: probe}}
        trait_type: Can be a single trait type (str) or None for multi-intervention
        target_vector: Can be a single target vector or a dict {trait_type: target_vector} for multi-intervention
        N: Intervention strength (can be a single value or dict {trait_type: N})
        from_layer: Starting layer
        to_layer: Ending layer
    
    Returns a function that can be used with TraceDict
    """
    # Check if we have multiple interventions
    is_multi_intervention = (isinstance(control_probe_dict, dict) and 
                            len(control_probe_dict) > 0 and 
                            isinstance(next(iter(control_probe_dict.values())), dict) and
                            next(iter(control_probe_dict.keys())) in ["goal_persistence", "independence", "rigidity"])
    
    if is_multi_intervention:
        # Multiple interventions: apply all of them sequentially
        # target_vector should be a dict {trait_type: target_vector}
        target_vectors = target_vector if isinstance(target_vector, dict) else {}
        N_dict = N if isinstance(N, dict) else {trait: N for trait in control_probe_dict.keys()}
        
        def edit_inter_rep_multi_layers(output, layer_name):
            if "model.layers." not in layer_name:
                return output
            
            try:
                layer_num = int(layer_name[layer_name.rfind("model.layers.") + len("model.layers."):])
            except ValueError:
                return output
            
            # Only intervene in specified layer range
            if not (from_layer <= layer_num < to_layer):
                return output
            
            # Get the last token's hidden state
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Handle different tensor shapes
            if len(hidden_states.shape) == 3:
                cloned_inter_rep = hidden_states[:, -1, :].unsqueeze(0).detach().clone().to(torch.float)
            elif len(hidden_states.shape) == 2:
                cloned_inter_rep = hidden_states.detach().clone().to(torch.float)
            else:
                return output
            
            # Apply all interventions sequentially
            probe_layer = layer_num + 1
            for trait_type_key, trait_probe_dict in control_probe_dict.items():
                if probe_layer in trait_probe_dict and trait_type_key in target_vectors:
                    probe = trait_probe_dict[probe_layer]
                    target_vec = target_vectors[trait_type_key]
                    N_val = N_dict.get(trait_type_key, N if not isinstance(N, dict) else 8)
                    
                    # Apply intervention for this trait
                    with torch.enable_grad():
                        cloned_inter_rep = optimize_intervention_rep(
                            cloned_inter_rep, 
                            target_vec, 
                            probe,
                            N=N_val,
                            normalized=False
                        )
            
            # Update the output
            if len(hidden_states.shape) == 3:
                hidden_states[:, -1, :] = cloned_inter_rep[0].to(torch.float16)
            elif len(hidden_states.shape) == 2:
                hidden_states = cloned_inter_rep.to(torch.float16)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return edit_inter_rep_multi_layers
    else:
        # Single intervention (backward compatibility)
        def edit_inter_rep_multi_layers(output, layer_name):
            """
            Edit intermediate representation at multiple layers using control probes
            
            This function is called by TraceDict during forward pass
            """
            if "model.layers." not in layer_name:
                return output
            
            # Extract layer number (for residual stream, layer name is just the number)
            try:
                layer_num = int(layer_name[layer_name.rfind("model.layers.") + len("model.layers."):])
            except ValueError:
                return output
            
            # Only intervene in specified layer range
            if not (from_layer <= layer_num < to_layer):
                return output
            
            # Get probe for this layer (probes are indexed by layer_num + 1 in some setups)
            probe_layer = layer_num + 1
            if probe_layer not in control_probe_dict:
                return output
            
            probe = control_probe_dict[probe_layer]
            
            # Get the last token's hidden state
            # For residual stream, output is typically a tuple where output[0] is the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Handle different tensor shapes
            if len(hidden_states.shape) == 3:
                # [batch, seq_len, hidden_dim] - get last token
                cloned_inter_rep = hidden_states[:, -1, :].unsqueeze(0).detach().clone().to(torch.float)
            elif len(hidden_states.shape) == 2:
                # [batch, hidden_dim] - use as is
                cloned_inter_rep = hidden_states.detach().clone().to(torch.float)
            else:
                return output
            
            # Apply intervention
            with torch.enable_grad():
                cloned_inter_rep = optimize_intervention_rep(
                    cloned_inter_rep, 
                    target_vector, 
                    probe,
                    N=N,
                    normalized=False
                )
            
            # Update the output
            if len(hidden_states.shape) == 3:
                hidden_states[:, -1, :] = cloned_inter_rep[0].to(torch.float16)
            elif len(hidden_states.shape) == 2:
                hidden_states = cloned_inter_rep.to(torch.float16)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        return edit_inter_rep_multi_layers


def generate_with_probes(
    model, 
    tokenizer, 
    messages, 
    reading_probe_dict=None,
    control_probe_dict=None,
    trait_type=None,
    target_vector=None,
    intervention_strength=8,  # Can be int or dict {trait_type: strength}
    from_layer=20,
    to_layer=30,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    device="cuda"
):
    """
    Generate response with optional reading and control probes
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        messages: List of message dicts with 'role' and 'content'
        reading_probe_dict: Dictionary of reading probes for activation detection
        control_probe_dict: Dictionary of control probes for intervention
        trait_type: Type of behavioral trait being controlled
        target_vector: Target vector for intervention [1, num_classes]
        intervention_strength: Strength of intervention (N parameter)
        from_layer: Starting layer for intervention
        to_layer: Ending layer for intervention
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run on
    
    Returns:
        Dictionary with:
            - response: Generated text
            - activations: Detected activations at each layer (if reading_probe_dict provided)
    """
    # Format prompt
    formatted_prompt = llama_v2_prompt(messages)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True).to(device)
    
    activations = {}
    
    # Determine which layers to trace (for residual stream, we trace the layer outputs)
    modified_layer_names = []
    # Check if we have multiple interventions
    is_multi_intervention = (control_probe_dict is not None and 
                            isinstance(control_probe_dict, dict) and 
                            len(control_probe_dict) > 0 and 
                            isinstance(next(iter(control_probe_dict.values())), dict) and
                            next(iter(control_probe_dict.keys())) in ["goal_persistence", "independence", "rigidity"])
    
    if control_probe_dict is not None and target_vector is not None:
        # Get all layers in the intervention range
        for layer_idx in range(from_layer, to_layer):
            layer_name = f"model.layers.{layer_idx}"
            
            # Check if any probe dict has a probe for this layer
            has_probe = False
            if is_multi_intervention:
                # Check all trait probe dicts
                for trait_probe_dict in control_probe_dict.values():
                    probe_layer = layer_idx + 1
                    if probe_layer in trait_probe_dict:
                        has_probe = True
                        break
            else:
                # Single intervention
                probe_layer = layer_idx + 1
                if probe_layer in control_probe_dict:
                    has_probe = True
            
            # Verify the layer exists in the model and add it
            if has_probe:
                for mod_name, mod in model.named_modules():
                    if mod_name == layer_name:
                        modified_layer_names.append(mod_name)
                        break
    
    # Create intervention function
    edit_function = None
    if control_probe_dict is not None and target_vector is not None:
        # Check if we have multiple interventions
        is_multi_intervention = (isinstance(control_probe_dict, dict) and 
                                len(control_probe_dict) > 0 and 
                                isinstance(next(iter(control_probe_dict.values())), dict) and
                                next(iter(control_probe_dict.keys())) in ["goal_persistence", "independence", "rigidity"])
        
        if is_multi_intervention:
            # For multi-intervention, trait_type is None and target_vector is a dict
            edit_function = create_edit_function(
                control_probe_dict,
                None,  # trait_type is None for multi-intervention
                target_vector,  # dict {trait_type: target_vector}
                N=intervention_strength,  # can be dict or single value
                from_layer=from_layer,
                to_layer=to_layer
            )
        else:
            # Single intervention (backward compatibility)
            edit_function = create_edit_function(
                control_probe_dict,
                trait_type,
                target_vector,
                N=intervention_strength,
                from_layer=from_layer,
                to_layer=to_layer
            )
    
    # Generate with or without intervention
    with torch.no_grad():
        if modified_layer_names and edit_function is not None:
            # Use TraceDict for intervention
            with TraceDict(model, modified_layer_names, edit_output=edit_function) as ret:
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id
                )
        else:
            # Standard generation
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id
            )
    
    # Decode response
    full_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in full_text:
        response = full_text.split("[/INST]")[-1].strip()
    else:
        response = full_text[len(formatted_prompt):].strip()
    
    # Detect activations if reading probes provided
    # We need to run a forward pass to get hidden states at the last position
    if reading_probe_dict is not None:
        # Get hidden states from the model at the last position of the input
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Check if reading_probe_dict is a dict of dicts (multiple traits) or single dict
        # If first key maps to a dict, it's multiple traits
        is_multi_trait = (isinstance(reading_probe_dict, dict) and 
                         len(reading_probe_dict) > 0 and 
                         isinstance(next(iter(reading_probe_dict.values())), dict))
        
        if is_multi_trait:
            # Multiple traits: reading_probe_dict = {"goal_persistence": {layer: probe}, ...}
            for trait_type, trait_probe_dict in reading_probe_dict.items():
                if trait_probe_dict:  # Only process if probes are loaded for this trait
                    trait_activations = {}
                    for layer_num in trait_probe_dict.keys():
                        if layer_num < len(hidden_states):
                            act_info = detect_activation(
                                trait_probe_dict, 
                                hidden_states[layer_num], 
                                layer_num
                            )
                            if act_info:
                                trait_activations[layer_num] = act_info
                    if trait_activations:
                        activations[trait_type] = trait_activations
        else:
            # Single trait: reading_probe_dict = {layer: probe}
            for layer_num in reading_probe_dict.keys():
                if layer_num < len(hidden_states):
                    act_info = detect_activation(
                        reading_probe_dict, 
                        hidden_states[layer_num], 
                        layer_num
                    )
                    if act_info:
                        activations[layer_num] = act_info
    
    return {
        "response": response,
        "activations": activations,
        "full_text": full_text
    }

