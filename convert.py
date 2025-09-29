from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from llava.model.builder import load_pretrained_model
import torch.onnx
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def show_image_in_plt(image):
    image = image.permute((1, 2, 0)).to(torch.float32).cpu()

    print('image: ', image)
    print('image shape: ', image.shape)
    plt.imshow(image)
    plt.show()

do_inf = True
convert_to_onnx = True 

if do_inf:
    tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-7b", model_name="llava-v1.5-13b", model_base=None, load_8bit=False, load_4bit=False, attn_implementation="eager" )

    print_model_arch = True
    if print_model_arch:
        for name, module in model.named_modules():
            # 'name' is the string path (e.g., 'feature_extractor.0')
            # 'module' is the actual PyTorch layer object (e.g., Conv2d)
            
            # Skip the top-level container itself
            if name == '':
                continue
                
            print(f"Layer Name: {name:<25} | Type: {type(module).__name__}")
            
            for weight_name, weight_param in module.named_parameters():
                print(f"Layer: {weight_name:<25} | Shape: {list(weight_param.shape)} | datatype: {weight_param.dtype}")

    #prompt = "do you know about high performance computing collectives, tell me about each of these collectives."
    prompt = "what is inside the picutre [IMG]?"


    input_ids = torch.tensor(tokenizer(text=prompt)['input_ids'], dtype=torch.int32).to(device='cuda')
    print('input_ids: ', input_ids)

    sample_images = ['all-gather.png', 'all-to-all.png', 'all-reduce.png', 'broadcast.png', 'reduce.png', 'reduce-scatter.png', 'scatter.png']

    Resize = transforms.Resize((336, 336))
    ToTensor = transforms.ToTensor()
    images = []
    for image in sample_images:
        image = ToTensor(Image.open(os.path.join('images', image)).convert('RGB')).to(dtype=torch.float16).to(device='cuda').unsqueeze(0)
        downsampled_image = F.interpolate(image, scale_factor=0.5, mode='bilinear')
        resized_image = Resize(downsampled_image).squeeze(0)

        images.append(resized_image)

    input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    example_inputs = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images)
    # attention mask and position ids are swapped compared to modeling llama forward argument order
    example_inputs = (example_inputs[0], example_inputs[2], example_inputs[1], *example_inputs[3:])
    print('example_inputs: ', example_inputs)

    if convert_to_onnx:
        onnx_program = torch.onnx.export(model, example_inputs, f='llava.onnx')
    #print(tokenizer.batch_decode(model.generate(inputs=input_ids, images=images)))

    

