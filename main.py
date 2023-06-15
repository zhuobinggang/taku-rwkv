import os
from functools import lru_cache
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv.model import RWKV  # dynamic import to make RWKV_CUDA_ON work
from my_pipline import PIPELINE, PIPELINE_ARGS

@lru_cache(maxsize=None)
def get_args():
    args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    return args


@lru_cache(maxsize=None)
def get_pipeline():
    path = '/home/taku/research/LANGUAGE_MODELS/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth' 
    model = RWKV(model=path, strategy='cuda fp16i8')
    pipeline = PIPELINE(model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
    return pipeline
    
def send(txt):
    response = get_pipeline().generate(txt, token_count=200, args=get_args())
    return response
