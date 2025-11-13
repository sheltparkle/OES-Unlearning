from torchvision import datasets
from parameter import *
from data_utils import *
from torch.utils.data import Subset
from function import train_model
from torchvision import transforms


if __name__ == '__main__':
    device = "cuda:0"
    epochs = 100
    lr = 0.0004
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    save_path = PATH_CRETRAIN

    # Model and model name selection
    model_name = "ResNet18"  # AllCNN & ResNet18 & ResNet50
    model = MODEL_CLASSES[model_name]()  # Create a model instance based on the model name

    training_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=False, transform=transform, download=True)

    retain_training_dataset = select_samples_by_class(training_dataset, target_classes=RETAIN_CLASSES)
    retain_test_dataset = select_samples_by_class(test_dataset, target_classes=RETAIN_CLASSES)

    train_loader = torch.utils.data.DataLoader(retain_training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(retain_test_dataset, batch_size=BATCH_SIZE*4, shuffle=False)
    train_model(model, model_name, retain_training_dataset, device, epochs, lr, grad_clip, weight_decay, opt_func, train_loader,
                test_loader, save_path)
