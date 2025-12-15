import torch
import numpy as np
from stable_baselines3 import PPO

def flip_nature_cnn_weights(model_path, save_path):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # Access policy's state dict (the weights)
    policy = model.policy
    
    with torch.no_grad():
        # 1. Flip Convolutional Kernels (eyes)
        # NatureCNN has 3 Conv layers. We flip the weights width-wise (last dim). 
        # This makes the filters recognize mirrored features.
        print("Flipping Convolutional layers...")
        policy.features_extractor.cnn[0].weight.data = policy.features_extractor.cnn[0].weight.data.flip(-1)
        policy.features_extractor.cnn[2].weight.data = policy.features_extractor.cnn[2].weight.data.flip(-1)
        policy.features_extractor.cnn[4].weight.data = policy.features_extractor.cnn[4].weight.data.flip(-1)

        # 2. Reorder first Linear Layer 
        # The output of the CNN is flattened before hitting the linear layer.
        # We need to mirror the incoming connections to match the flipped image.
        
        # Standard Atari NatureCNN Dimensions:
        # Input: 84x84
        # Conv1 Output: 20x20 (64 channels)
        # Final Conv Output: 64 channels x 7 height x 7 width
        # Flattened Size: 64 * 7 * 7 = 3136
        
        print("Reordering Linear Layer weights...")
        
        # Get the weight matrix of the first linear layer after the CNN
        # In SB3 NatureCNN, the flattened output goes into 'linear.0' inside features_extractor
        
        linear_layer = policy.features_extractor.linear[0]
        weights = linear_layer.weight.data # Shape: (512, 3136)
        
        # We need to reshape, flip width, and flatten back.
        
        # Standard Atari (84x84) NatureCNN
        n_channels = 64
        h = 7
        w = 7
        
        # Verify shape matches expectation
        expected_size = n_channels * h * w
        current_size = weights.shape[1]
        
        if current_size != expected_size:
            print(f"WARNING: Unexpected feature size {current_size}. Expected {expected_size}.")
            print("Skipping Linear flip (Model might act weird). Check architecture.")
        else:
            # Reshape to (Out_Features, Channels, Height, Width)
            reshaped_weights = weights.view(weights.shape[0], n_channels, h, w)
            
            # Flip along the Width dimension (axis 3)
            flipped_weights = reshaped_weights.flip(3)
            
            # Flatten back to (Out_Features, Flattened_Input)
            linear_layer.weight.data = flipped_weights.flatten(start_dim=1)
            print("Linear layer reordered successfully.")

    # 3. Save
    print(f"Saving flipped model to {save_path}...")
    model.save(save_path)
    print("Done! You can now use this zip file as the Left Agent.")

if __name__ == "__main__":
    SOURCE_MODEL = "right_paddle_20mill/models/best_model.zip"
    TARGET_MODEL = "right_paddle_20mill/pong_left_flipped_20mill.zip"
    
    flip_nature_cnn_weights(SOURCE_MODEL, TARGET_MODEL)