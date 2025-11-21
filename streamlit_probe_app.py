"""
Streamlit app for interactive probe-based inference
"""
import streamlit as st
import torch
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from probe_inference_utils import (
    load_model_and_tokenizer,
    load_probe_dict,
    generate_with_probes,
    BEHAVIORAL_TRAIT_LABELS,
    BEHAVIORAL_TRAIT_NAMES
)

# Page config
st.set_page_config(
    page_title="Probe-Based Chat Interface",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "reading_probes" not in st.session_state:
    st.session_state.reading_probes = {}
if "control_probes" not in st.session_state:
    st.session_state.control_probes = {}
if "probe_dir" not in st.session_state:
    st.session_state.probe_dir = None
if "trait_type" not in st.session_state:
    st.session_state.trait_type = "goal_persistence"


@st.cache_resource
def load_model_cached():
    """Cached model loading"""
    try:
        # Token will be read from HF_TOKEN environment variable
        model, tokenizer = load_model_and_tokenizer(access_token=None)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_resource
def load_probes_cached(probe_dir, probe_type, trait_type):
    """Cached probe loading"""
    try:
        probe_path = os.path.join(probe_dir, probe_type)
        if not os.path.exists(probe_path):
            return {}
        return load_probe_dict(probe_path, trait_type)
    except Exception as e:
        st.error(f"Error loading {probe_type} probes: {e}")
        return {}


def main():
    st.title("Probe-Based Chat Interface")
    st.markdown("Interactive chat with behavioral trait detection and control")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Probe directory
        default_probe_dir = "output/20251121_084552_1000_per_subcategory_gpt5.1_result/probe_checkpoints"
        probe_dir = st.text_input(
            "Probe Directory",
            value=default_probe_dir,
            help="Path to probe checkpoints folder"
        )
        
        if probe_dir != st.session_state.probe_dir:
            st.session_state.probe_dir = probe_dir
            st.session_state.reading_probes = {}
            st.session_state.control_probes = {}
        
        # Trait type selection
        trait_type = st.selectbox(
            "Behavioral Trait",
            options=["goal_persistence", "independence", "rigidity"],
            index=0,
            help="Select which behavioral trait to detect/control"
        )
        
        if trait_type != st.session_state.trait_type:
            st.session_state.trait_type = trait_type
            st.session_state.reading_probes = {}
            st.session_state.control_probes = {}
        
        st.divider()
        
        # Load model button
        if st.button("Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                model, tokenizer = load_model_cached()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
        
        # Load probes button
        if st.button("Load Probes", use_container_width=True):
            if not os.path.exists(probe_dir):
                st.error(f"Probe directory not found: {probe_dir}")
            else:
                with st.spinner("Loading probes..."):
                    reading_probes = load_probes_cached(probe_dir, "reading_probe", trait_type)
                    control_probes = load_probes_cached(probe_dir, "control_probe", trait_type)
                    
                    st.session_state.reading_probes = reading_probes
                    st.session_state.control_probes = control_probes
                    
                    st.success(f"Loaded {len(reading_probes)} reading probes and {len(control_probes)} control probes")
        
        st.divider()
        
        # Intervention settings
        st.header("Intervention Settings")
        
        enable_intervention = st.checkbox(
            "Enable Intervention",
            value=False,
            help="Use control probes to modify model behavior"
        )
        
        if enable_intervention:
            trait_names = BEHAVIORAL_TRAIT_NAMES.get(trait_type, ["Low", "Medium", "High"])
            target_level = st.selectbox(
                "Target Level",
                options=[0, 1, 2],
                format_func=lambda x: trait_names[x],
                help="Target behavioral trait level to steer towards"
            )
            
            intervention_strength = st.slider(
                "Intervention Strength",
                min_value=1,
                max_value=20,
                value=8,
                help="Strength of the intervention (N parameter)"
            )
            
            from_layer = st.slider(
                "From Layer",
                min_value=0,
                max_value=40,
                value=20,
                help="Starting layer for intervention"
            )
            
            to_layer = st.slider(
                "To Layer",
                min_value=0,
                max_value=40,
                value=30,
                help="Ending layer for intervention"
            )
        else:
            target_level = None
            intervention_strength = 8
            from_layer = 20
            to_layer = 30
        
        st.divider()
        
        # Generation settings
        st.header("Generation Settings")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=512,
            value=256
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
        
        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05
        )
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show activations if available
                if "activations" in message and message["activations"]:
                    with st.expander("View Activations"):
                        show_activations(message["activations"], trait_type)
        
        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if model is loaded
            if st.session_state.model is None or st.session_state.tokenizer is None:
                st.error("Please load the model first in the sidebar!")
                return
            
            # Prepare messages for generation
            messages = []
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Prepare intervention parameters
            target_vector = None
            if enable_intervention:
                if not st.session_state.control_probes:
                    st.warning("Control probes not loaded. Please load probes first.")
                else:
                    # Create target vector (one-hot)
                    num_classes = len(BEHAVIORAL_TRAIT_LABELS[trait_type])
                    target_vector = torch.zeros(1, num_classes)
                    target_vector[0, target_level] = 1.0
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = generate_with_probes(
                            model=st.session_state.model,
                            tokenizer=st.session_state.tokenizer,
                            messages=messages,
                            reading_probe_dict=st.session_state.reading_probes if st.session_state.reading_probes else None,
                            control_probe_dict=st.session_state.control_probes if enable_intervention else None,
                            trait_type=trait_type if enable_intervention else None,
                            target_vector=target_vector,
                            intervention_strength=intervention_strength if enable_intervention else 0,
                            from_layer=from_layer,
                            to_layer=to_layer,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        
                        response = result["response"]
                        activations = result.get("activations", {})
                        
                        st.markdown(response)
                        
                        # Store message with activations
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "activations": activations
                        })
                        
                        # Show activations inline
                        if activations:
                            with st.expander("View Activations"):
                                show_activations(activations, trait_type)
                    
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        st.header("Activation Analysis")
        
        if st.session_state.messages:
            # Collect all activations from chat history
            all_activations = {}
            for msg in st.session_state.messages:
                if "activations" in msg and msg["activations"]:
                    for layer, act_info in msg["activations"].items():
                        if layer not in all_activations:
                            all_activations[layer] = []
                        all_activations[layer].append(act_info)
            
            if all_activations:
                # Show activation summary
                st.subheader("Layer-wise Activations")
                
                # Create summary dataframe
                layer_data = []
                for layer, acts in all_activations.items():
                    avg_probs = np.mean([a["probabilities"] for a in acts], axis=0)
                    avg_conf = np.mean([a["confidence"] for a in acts])
                    pred_class = int(np.round(np.mean([a["predicted_class"] for a in acts])))
                    
                    layer_data.append({
                        "Layer": layer,
                        "Predicted": BEHAVIORAL_TRAIT_NAMES[trait_type][pred_class],
                        "Confidence": f"{avg_conf:.2%}",
                        "Low": f"{avg_probs[0]:.2%}",
                        "Medium": f"{avg_probs[1]:.2%}",
                        "High": f"{avg_probs[2]:.2%}"
                    })
                
                df = pd.DataFrame(layer_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                if len(all_activations) > 0:
                    st.subheader("Activation Heatmap")
                    plot_activation_heatmap(all_activations, trait_type)
            else:
                st.info("No activations detected yet. Send a message to see activations.")
        else:
            st.info("Start a conversation to see activation analysis.")


def show_activations(activations, trait_type):
    """Display activation information for a message"""
    if not activations:
        st.info("No activations available")
        return
    
    trait_names = BEHAVIORAL_TRAIT_NAMES.get(trait_type, ["Low", "Medium", "High"])
    
    for layer, act_info in sorted(activations.items()):
        with st.expander(f"Layer {layer}"):
            pred_class = act_info["predicted_class"]
            probs = act_info["probabilities"]
            confidence = act_info["confidence"]
            
            st.write(f"**Predicted:** {trait_names[pred_class]} (confidence: {confidence:.2%})")
            
            # Probability bars
            for i, (prob, name) in enumerate(zip(probs, trait_names)):
                st.progress(prob, text=f"{name}: {prob:.2%}")


def plot_activation_heatmap(all_activations, trait_type):
    """Create a heatmap of activations across layers"""
    layers = sorted(all_activations.keys())
    trait_names = BEHAVIORAL_TRAIT_NAMES.get(trait_type, ["Low", "Medium", "High"])
    
    # Prepare data
    heatmap_data = []
    for layer in layers:
        acts = all_activations[layer]
        avg_probs = np.mean([a["probabilities"] for a in acts], axis=0)
        heatmap_data.append(avg_probs)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.T,
        x=[f"Layer {l}" for l in layers],
        y=trait_names,
        colorscale='Viridis',
        text=[[f"{val:.2%}" for val in row] for row in heatmap_data.T],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title="Activation Probabilities Across Layers",
        xaxis_title="Layer",
        yaxis_title="Trait Level",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

