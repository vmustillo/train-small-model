import torch
from train import LanguageModel, decode, device

def load_previous_model_weights(device, path='mac_coder_model.pth'):
    model = LanguageModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval() # Ready to generate!

    return model

if __name__ == "__main__":
    model = load_previous_model_weights(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
