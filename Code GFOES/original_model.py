from torchvision import datasets
from parameter import *
from data_utils import *
from function import train_model
from torch.utils.data import Subset
from torchvision import transforms


if __name__ == '__main__':
    device = "cuda:0"
    epochs = 20
    lr = 0.0004
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



    save_path = PATH_ORIGINAL

    model_name = "ResNet18"  # AllCNN & ResNet18 & ResNet50
    model = MODEL_CLASSES[model_name]()

# cifar10
    training_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATASET_ROOT, train=False, transform=transform, download=True)


    train_loader = torch.utils.data.DataLoader(training_dataset , batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE * 4, shuffle=False)

    # 调用训练函数
    train_model(model, model_name, training_dataset, device, 50, lr, grad_clip, weight_decay, opt_func,
                train_loader, test_loader, save_path)

