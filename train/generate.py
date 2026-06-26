def load_previous_model_weights(model, device, path):
    model = LanguageModel().to(device)
    model.load_state_dict(torch.load('mac_coder_model.pth'))
    model.eval() # Ready to generate!

    return model