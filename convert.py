from llava.model.builder import load_pretrained_model
import torch.onnx

tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-7b", model_name="llava-v1.5-13b", model_base=None, load_8bit=False, load_4bit=False)

print_model_arch = False
if print_model_arch:
    for name, module in model.named_modules():
        # 'name' is the string path (e.g., 'feature_extractor.0')
        # 'module' is the actual PyTorch layer object (e.g., Conv2d)
        
        # Skip the top-level container itself
        if name == '':
            continue 
            
        print(f"Layer Name: {name:<25} | Type: {type(module).__name__}")

prompt = "Can you give me a brief summary of Hitchhiker's guide to the galaxy?"


input_ids = torch.tensor(tokenizer(text=prompt)['input_ids'], dtype=torch.int32).to(device='cuda')
print('input_ids: ', input_ids)


print(tokenizer.batch_decode(model.generate(inputs=torch.unsqueeze(input_ids, 0))))

