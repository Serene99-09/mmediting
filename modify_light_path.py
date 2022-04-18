import torch

path = "exp6_iter_200000.pth"
stuff = torch.load(path)
to_rm = ['generator.conv_body.weight', 'generator.conv_body.bias', 
'generator.conv_up1.weight', 'generator.conv_up1.bias', 'generator.conv_up2.weight', 
'generator.conv_up2.bias', 'generator_ema.conv_body.weight', 
'generator_ema.conv_body.bias', 'generator_ema.conv_up1.weight', 
'generator_ema.conv_up1.bias', 'generator_ema.conv_up2.weight', 
'generator_ema.conv_up2.bias']
for m in to_rm:
    stuff['state_dict'].pop(m)
print(stuff['state_dict'].keys())
torch.save(stuff, "exp6_light_path.pth")