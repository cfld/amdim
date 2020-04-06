import re
import torch
from torch import nn
from torch.nn import functional as F

from model import Encoder
from ben_dataset import BENTransformValid
from torch.hub import load_state_dict_from_url

def to_numpy(x):
    return x.cpu().numpy()

class AMDIMEncoder(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        
        config = {
            "ndf"          : 128,
            "num_channels" : 12,
            "n_rkhs"       : 1024,
            "n_depth"      : 3,
            "encoder_size" : 128,
            "use_bn"       : 0,
        }
        
        dummy_batch = torch.zeros((2, config['num_channels'], config['encoder_size'], config['encoder_size']))
        
        self.encoder = Encoder(dummy_batch, **config)
        
        state_dict = {k:v for k,v in state_dict.items() if 'encoder.' in k}
        state_dict = {k.replace('encoder.', '').replace('module.', ''):v for k,v in state_dict.items()}
        self.encoder.load_state_dict(state_dict)
        
        self.transform = BENTransformValid()
    
    def forward(self, x):
        assert len(x.shape) == 4, "Input must be (batch_size, 12, 128, 128)"
        assert x.shape[1] == 12, "Input must be (batch_size, 12, 128, 128)"
        assert x.shape[2] == 128, "Input must be (batch_size, 12, 128, 128)"
        assert x.shape[3] == 128, "Input must be (batch_size, 12, 128, 128)"
        
        # --
        # Preprocessing
        
        device = x.device
        x = x.cpu()
        
        tmp = [xx.numpy().transpose(1, 2, 0) for xx in x]
        tmp = [self.transform(xx) for xx in tmp]
        x   = torch.stack(tmp)
        
        x = x.to(device)
        
        # --
        # Forward
        acts = self.encoder._forward_acts(x)
        out  = self.encoder.rkhs_block_1(acts[self.encoder.dim2layer[1]])
        out  = out[:,:,0,0]
        
        return out


def amdim_encoder(pretrained=True, progress=True, model_url='tmp.pth'):
    assert pretrained == True
    state_dict = load_state_dict_from_url(model_url)
    return AMDIMEncoder(state_dict)


if __name__ == "__main__":
    _ = torch.manual_seed(123)
    
    model = amdim_encoder()
    x = 100 * torch.rand((10, 12, 128, 128))
    print(float((model(x) ** 2).sum()))
