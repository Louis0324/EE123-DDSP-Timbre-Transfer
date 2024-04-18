import torch
import yaml
from vocoder import Vocoder
from my_utils import calc_nparam

def load_model(yaml_name, device):
    # load yaml config
    with open(yaml_name, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    # define model
    print('Loading model...')
    model = Vocoder(
        hidden_dim=config['hidden_dim'], 
        nharmonics=config['nharmonics'], 
        nbands=config['nbands'], 
        attenuate=config['attenuate'], 
        fs=config['fs'], 
        framesize=config['framesize'], 
        temperature=config['temperature'],
        dilations=config['dilations'], 
        nstacks=config['nstacks'],
        use_harmonic_conv=config['use_harmonic_conv'],
        reverb_len=config['reverb_len']
    )
    
    # load checkpoint
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
    model = model.to(device)
    print(f'number of parameters: {calc_nparam(model)}')
    return model