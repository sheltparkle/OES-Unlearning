import torch
from torch import nn
import torch.nn.functional as F

def evaluate(model, val_loader, device): 
    model.eval()  # Set the model to evaluation mode
    outputs = []
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():  # Disable gradient calculation for evaluation
            out = model(images)
            loss = F.cross_entropy(out, labels)
            _, preds = torch.max(out, dim=1)  # Get predictions
            correct = torch.sum(preds == labels).item()  # Count correct predictions
            total_correct += correct
            total_samples += labels.size(0)

        outputs.append({'Loss': loss.item(), 'Acc': correct / labels.size(0)})

    avg_loss = sum(x['Loss'] for x in outputs) / len(outputs)  # Calculate average loss
    avg_acc = total_correct / total_samples  # Calculate average accuracy

    return {'Loss': avg_loss, 'Acc': avg_acc}





def train_model(model, model_name, dataset, device, epochs, lr, grad_clip, weight_decay, opt_func, train_loader,
                test_loader, save_path):

    model.to(device)  # Move the model to the specified device
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)  # Initialize optimizer

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = F.cross_entropy(out, labels)  # Calculate loss
            loss.backward()  # Backpropagate the loss
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)  # Clip gradients if necessary
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients

        # Evaluate the model on the validation set after each epoch
        val_result = evaluate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_result["Loss"]}, Val Acc: {val_result["Acc"]}')

    # Save the trained model
    torch.save(model.state_dict(), f"{save_path}/{model_name}_{dataset.__class__.__name__}.pth")



