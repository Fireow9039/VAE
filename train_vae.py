import random
import torch
import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import optim
from MyVAE import MyVAE

# Crop proportions for the environment image to remove unnecessary parts
crop_proportions = (0.4, 0.0, 1.0, 1.0)
img_dim = (64, 64)  # Target image dimensions for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vae():
    # Initialize the gym environment with rendering mode set
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    Ang_minimum = 0.26
    Ang_maximum = 0.31

    # Get the initial observation and image rendering
    obs = env.reset()
    img = env.render()
    crop_dim = (
        int(crop_proportions[0] * img.shape[0]),
        int(crop_proportions[1] * img.shape[1]),
        int(crop_proportions[2] * img.shape[0]),
        int(crop_proportions[3] * img.shape[1])
    )

    # VAE model and parameters
    input_channels = 3
    latent_dim = 10
    training_size = 2000
    batch_size = latent_dim * 10
    n_epochs = 400

    # Initialize VAE and optimizer
    vae = MyVAE(in_channels=input_channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    imgs = np.zeros((training_size, input_channels, *img_dim), dtype=np.float32)

    # Collect pixel data from the gym environment
    frame_idx = 0
    i = 0  # Counter for valid training images
    while i < training_size:
        frame_idx += 1
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Get pixel observations, crop, and resize if within angle limits
        angle = obs[2]
        if Ang_minimum <= abs(angle) <= Ang_maximum:
            img = env.render()
            img = img[crop_dim[0]:crop_dim[2], crop_dim[1]:crop_dim[3], :]
            img = cv2.resize(img, dsize=img_dim, interpolation=cv2.INTER_CUBIC)
            img = img.swapaxes(0, 2).reshape((1, input_channels, *img_dim)).astype(np.float32) / 255.0
            imgs[i] = img
            i += 1  

        # Reset environment if the episode ends or after 20 frames
        if frame_idx >= 20:
            obs = env.reset()
            frame_idx = 0

    env.close()

    # Visualization setup
    plt.ion()
    plt.show()

    # Train VAE
    for epoch in range(n_epochs):
        start_idx = random.randint(0, training_size - batch_size)
        train_imgs = imgs[start_idx: start_idx + batch_size]
        out_imgs = vae(torch.from_numpy(train_imgs.copy()).to(device))
        loss = vae.loss(*out_imgs, kl_w=0.0005)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

        # Show generated image periodically
        if (epoch + 1) % 10 == 0:
            rand_idx = np.random.randint(0, batch_size - 1)
            im = out_imgs[0][rand_idx: rand_idx + 1].detach().cpu().numpy().reshape((1, 3, *img_dim)).swapaxes(1, 3)
            im = (im * 255.0).astype(np.uint8)
            plt.imshow(im[0])
            plt.axis('off')
            plt.show()
            plt.pause(0.1)

    # Save model and training visualization
    torch.save(vae.state_dict(), 'vae.pth')
    plt.savefig('vae_training.png')
    plt.show()

if __name__ == '__main__':
    train_vae()
