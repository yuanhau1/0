import torch
from diffusers import FluxPipeLine
cpu_or_not = input("Cpu? ")
prompt = input('Prompt: ')
save_name = input("File Name: ")
if cpu_or_not = "yes":
  pipe.enable_model_cpu_offload()
if cpu_or_not = "no":
  pipe.disable_model_cpu_offload()
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
    )
image.save(save_name)
# THIS IS ONLY TRANSFER ONLY!
