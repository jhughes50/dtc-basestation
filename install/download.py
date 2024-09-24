#!/usr/bin/env python3
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)

disable_torch_init()
model_name = get_model_name_from_path("llava-onevision-qwen2-7b-ov-chat/")

load_pretrained_model(
    "llava-onevision-qwen2-7b-ov-chat/",
    None,
    model_name,
    load_8bit=False,
    load_4bit=False,
)
