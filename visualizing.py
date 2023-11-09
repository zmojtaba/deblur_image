import matplotlib.pyplot as plt 
import torch

def visualizing_data(predicted, data, label, image_size, predicted_image_size, epoch):
    plt.figure(figsize=(15,15),tight_layout = True)

    plt.subplot(1,3,1)
    plt.imshow(predicted[0].to('cpu').detach().numpy().reshape(predicted_image_size, predicted_image_size, 3), cmap='gray')
    plt.title(f"predicted image")

    plt.subplot(1,3,2)
    plt.imshow(data[0].to('cpu').detach().numpy().reshape(image_size, image_size, 3), cmap='gray')
    plt.title(f"motion image")

    plt.subplot(1,3,3)
    plt.imshow(label[0].to('cpu').detach().numpy().reshape(image_size, image_size, 3), cmap='gray')
    plt.title(f"real image")


    plt.tight_layout()
    plt.axis("OFF")
    plt.savefig(f"project//stunmaster//improve_quality//my_model/results/{epoch}.png")

def saving_model(state , file_dir):
    torch.save( state, file_dir)
