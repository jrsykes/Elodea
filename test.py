#%%
import torch


a_tensor =  torch.tensor([[100, 134],[50,  67],[25,  34], [13,  17]]).cuda()

# move a_tensor to cpu

a_tensor = a_tensor.cpu()

print(a_tensor)
#%%
a_tensor_prod = torch.prod(a_tensor, 1)
a_tensor_prod_cum = a_tensor_prod.cumsum(0)[:-1].cuda()
a_tensor_prod_cum
#%%