
import os
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.neuron
import torch.nn.functional as F

# To use one neuron core per worker
os.environ["NEURON_RT_NUM_CORES"] = "1"

# saved weights name
AWS_NEURON_TRACED_WEIGHTS_NAME = "neuron_model.pt"

# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def model_fn(model_dir):
    # load tokenizer and neuron model from model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = torch.jit.load(os.path.join(model_dir, AWS_NEURON_TRACED_WEIGHTS_NAME))
    model_config = AutoConfig.from_pretrained(model_dir)

    return model, tokenizer, model_config


def predict_fn(data, model_tokenizer_model_config):
    # destruct model and tokenizer
    model, tokenizer, model_config = model_tokenizer_model_config
    
    # Tokenize sentences
    inputs = data.pop("inputs", data)
    encoded_input = tokenizer(
        inputs,
        return_tensors="pt",
        max_length=model_config.traced_sequence_length,
        padding="max_length",
        truncation=True,
    )
    # convert to tuple for neuron model
    neuron_inputs = tuple(encoded_input.values())

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(*neuron_inputs)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # return dictonary, which will be json serializable
    return {"vectors": sentence_embeddings[0].tolist()}
