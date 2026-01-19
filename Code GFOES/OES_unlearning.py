
from GFN import *
from parameter import *
import torch.nn.functional as F
unlearning_model = MODEL_CLASSES[model_name]().to(device=DEVICE)


unlearning_model .load_state_dict(torch.load(save_path))
optimizer_unlearning = torch.optim.Adam(unlearning_model.parameters(), lr=0.004, weight_decay=1e-4)


# Unlearning process
def erasure_phase():
    for target_class in FORGET_CLASSES:
        target_labels = torch.full((BATCH_SIZE,), target_class, device=DEVICE)  # Create target labels for the class to forget
        for epoch in range(1):  # Single epoch unlearning
            unlearning_model.train(True)  # Set the model to training mode
            for images, labels in sub_retain_training_dataset_loader :
                unlearning_model.train()  # Ensure the model is in training mode
                images, labels = images.to(DEVICE), labels.to(DEVICE)


                gafn_gen.load_state_dict(torch.load(path_gafn_genertor))
                gafn_gen.eval()  # Set GAFN to evaluation mode

                z = torch.randn(BATCH_SIZE, NZ, device=DEVICE)  # Generate random noise
                fake_images = gafn_gen(z)  # Generate fake images using the GAFN generator


                model_output_real = unlearning_model(images)
                model_output_fake = unlearning_model(fake_images)


                loss_real = F.cross_entropy(model_output_real, labels)
                loss_fake = F.cross_entropy(model_output_fake, target_labels)
                total_loss = loss_fake+loss_real

                # Backpropagate and update the model's weights
                optimizer_unlearning.zero_grad()
                total_loss.backward()
                optimizer_unlearning.step()





def repair_phase():
    optimizer_enhance = torch.optim.Adam(unlearning_model.parameters(), lr=0.0004, weight_decay=1e-4)
    for epoch in range(1):  # Single epoch enhancement
        unlearning_model.train(True)  # Set the model to training mode
        for images, labels in sub_retain_training_dataset_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = unlearning_model(images)
            loss = F.cross_entropy(outputs, labels)

            # Backpropagate and update the model's weights
            optimizer_enhance.zero_grad()
            loss.backward()
            optimizer_enhance.step()





if __name__ == '__main__':
    erasure_phase()
    repair_phase()
    torch.save(unlearning_model.state_dict(), f"./{PATH_UNLEARNING}/{model_name}_{training_dataset.__class__.__name__}.pth")


