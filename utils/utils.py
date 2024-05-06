import sys
import os

os.environ['HF_HOME'] = "/root/hf_models/"
sys.path.insert(1, '/root/CS726/')
# sys.path.insert(1, '/raid/infolab/sarvam/elk_home_sarvam/pointer_decoding/utils')

# sys.path.insert(1, '/mnt/home/sarvam/pointer_decoding/')
# os.environ['HF_HOME'] = '/mnt/nas/sarvam/open_llm/huggingface_models/'


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from codes_constrained_llm import GPTBigCodeRATLM


from constants import *

model_mapping = {"codes_3b": "seeklhy/codes-3b-spider",
                 "phi3": "microsoft/Phi-3-mini-4k-instruct",
                 "codellama": "meta-llama/CodeLlama-7b-hf"}


def load_model(model_id=None, model_type=None, location=None):
    """
    Load the model (PointerDecodingLM or vanilla CausalLM and tokenizer with huggingface api

    Parameters:
    -----------
    model_id: str
        Model ID as used in transformers api

    model_type: str
        Model type (codes_3b, phi3_4k or codellama_7b)

    Returns:
    --------
    model: 
        Model object
    tokenizer: AutoTokenizer
        Tokenizer object

    """
    if model_type == "codes_3b":
        model_id = "seeklhy/codes-3b"
        if location is not None:
            model = GPTBigCodeRATLM.from_pretrained(location, torch_dtype="auto")
        else:
            model = GPTBigCodeRATLM.from_pretrained(model_id, torch_dtype="auto")

    elif model_type == "phi3_4k":
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def load_embed_model(id="google-bert/bert-large-uncased"):
    """
    Load the bert/codellama model and tokenizer

    Returns:
    --------
    model: BertForMaskedLM
        Model object
    tokenizer: AutoTokenizer
        Tokenizer object
    
    """
    model = AutoModelForMaskedLM.from_pretrained(id, torch_dtype="auto")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(id, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

