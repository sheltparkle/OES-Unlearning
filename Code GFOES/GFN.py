
from torchvision import datasets
import torch.nn.functional as F
from parameter import *
from model_utils import Generator
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_utils import select_samples_by_class


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


training_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=True, transform=transform, download=True)
dataset = select_samples_by_class(training_dataset, target_classes=RETAIN_CLASSES,samples_per_class=500)
sub_retain_training_dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



# Initialize the weights for the generator
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# # Initialize the generator and apply the weight initialization
gafn_gen = Generator(NZ, NGF, NC).to(DEVICE)
gafn_gen.apply(init_weights)

# Define the model name and save path
model_name = "ResNet18"  # Can be "AllCNN" & "ResNet18"& "ResNet18"
save_path = f"{PATH_ORIGINAL}/{model_name}_{training_dataset.__class__.__name__}.pth"

# Initialize save paths for the generator model
path_gafn_genertor =f"./{PATH_GAFN_GENERATOR}/{model_name}_{training_dataset.__class__.__name__}.pth"

# Load the original model and training model from saved states
original_model = MODEL_CLASSES[model_name]().to(device=DEVICE)
original_model.load_state_dict(torch.load(save_path))
train_model = MODEL_CLASSES[model_name]().to(device=DEVICE)
train_model.load_state_dict(torch.load(save_path))


if __name__ == '__main__':
    # Iterate over each target class to perform unlearning
    all_loss_fake_before = []
    all_loss_retain_train_model = []
    gafn_gen.apply(init_weights)
    optimizer_G = torch.optim.Adam(gafn_gen.parameters(), lr=0.004, weight_decay=1e-4)
    optimizer_train_model = torch.optim.Adam(train_model.parameters(), lr=0.004, weight_decay=1e-4)
    lamda = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True, device=DEVICE))
    num_forget = len(FORGET_CLASSES)

    target_onehot = F.one_hot(
        torch.tensor(FORGET_CLASSES, device=DEVICE), num_classes=N_CLASSES
    ).float()
    for epoch in range(20):
        for images, labels in sub_retain_training_dataset_loader:
            # ---------- 1. Maximize  ----------
            z = torch.randn(BATCH_SIZE * num_forget, NZ, device=DEVICE)
            fake_images = gafn_gen(z)
            logits_fake = original_model(fake_images)
            targets = target_onehot.repeat_interleave(BATCH_SIZE, dim=0)  # [B*F, C]
            loss_max = F.cross_entropy(logits_fake, targets).mean()

            # ---------- 2. Train model  ----------
            train_model.load_state_dict(torch.load(save_path))
            train_model.train()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss_real = F.cross_entropy(train_model(images), labels)
            total_loss = loss_max+loss_real
            optimizer_train_model.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer_train_model.step()

            # ---------- 3. Minimize & update G ----------
            loss_min = F.cross_entropy(train_model(images), labels)
            G_loss = (1 - lamda) * loss_min+ lamda * (1 / loss_max)
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            lamda.data += 0.01 * lamda.grad.data
            lamda.grad.zero_()
            lamda.data = torch.clamp(lamda.data, min=1e-5, max=0.9999)



            # Print the losses for each epoch
        print(
                f"[Epoch {epoch} "
                f"[loss_max: {loss_max.item()}] "
                f"[loss_min: {loss_min.item()}] "
                f"[total_loss: {G_loss.item()}] "
                f"[lamda:{lamda.item()}]"
        )

        # Save the generator model after each target class processing
    torch.save(gafn_gen.state_dict(), path_gafn_genertor)

