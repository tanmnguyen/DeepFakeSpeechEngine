import torch 

def load_state_dict(model, weight_path):
    """
    Load a model's state dictionary while handling size mismatches.

    Parameters:
        model (torch.nn.Module): The model to load the state dictionary into.
        weight_path (str): The path to the saved state dictionary.
    """
    # Load the state dictionary while ignoring size mismatches
    loaded_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

    # Get the current model state dictionary
    current_state_dict = model.state_dict()

    # Update the current state dictionary with parameters from the loaded state dictionary,
    # while ignoring any size mismatches
    for name, param in loaded_state_dict.items():
        if name in current_state_dict:
            if param.size() != current_state_dict[name].size():
                print(f"Ignoring size mismatch for parameter '{name}'")
                continue
            current_state_dict[name].copy_(param)

    # Load the updated state dictionary into the model
    model.load_state_dict(current_state_dict)

    return model 