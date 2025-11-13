import random
from torch.utils.data import Subset

def select_samples_by_class(dataset, target_classes=None, samples_per_class=None):

    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset does not have 'targets' or 'labels' attributes.")


    if target_classes is None:
        target_classes = list(set(labels))


    class_indices = {cls: [] for cls in target_classes}


    for idx, label in enumerate(labels):
        if label in target_classes:
            class_indices[label].append(idx)

    selected_indices = []
    for cls in target_classes:
        class_samples = class_indices[cls]

        if samples_per_class is None or samples_per_class > len(class_samples):
            selected_indices.extend(class_samples)
        else:
            selected_indices.extend(random.sample(class_samples, samples_per_class))


    subset_dataset = Subset(dataset, selected_indices)
    return subset_dataset