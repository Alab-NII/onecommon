from models.ctx_encoder import MlpContextEncoder, AttentionContextEncoder
from models.rnn_reference_model import RnnReferenceModel

MODELS = {
    'rnn_reference_model': RnnReferenceModel,
}

CTX_ENCODERS = {
    'mlp_encoder': MlpContextEncoder,
    'attn_encoder': AttentionContextEncoder,
}

def get_model_names():
    return MODELS.keys()


def get_model_type(name):
    return MODELS[name]

def get_ctx_encoder_names():
    return CTX_ENCODERS.keys()

def get_ctx_encoder_type(name):
    return CTX_ENCODERS[name]