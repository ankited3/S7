import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Create image augmentation pipeline with HorizontalFlip(10%) and ShiftScaleRotate(10%)
# and CoarseDropout(10%) with padding of 16px
base_transforms = A.Compose([
    A.HorizontalFlip(p=0.1),                             # 10% chance
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        border_mode=0,
        p=0.1                                           # 10% chance
    )
])

# Special CoarseDropout pipeline (applied conditionally)
coarse_dropout_pipeline = A.Compose([
    A.Pad(padding=16, fill=0, p=1.0),                   # Add 16px padding
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(16, 16),
        hole_width_range=(16, 16),
        fill=(0.485, 0.456, 0.406,),
        fill_mask=None,
        p=1.0  # Always apply if this pipeline runs
    ),
    A.Resize(32, 32, p=1.0)                             # Resize back to 32x32
])

normalization = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class CIFAR10WithCustomDropout(Dataset):
    def __init__(self, root, train=True, download=True, coarse_dropout_prob=0.1):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.coarse_dropout_prob = coarse_dropout_prob

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        image = np.array(image)

        # Apply base transforms (HorizontalFlip 30%, ShiftScaleRotate 20%)
        base_result = base_transforms(image=image)
        image = base_result['image']

        # Apply CoarseDropout with padding conditionally (10% chance)
        if np.random.random() < self.coarse_dropout_prob:
            coarse_dropout_result = coarse_dropout_pipeline(image=image)
            image = coarse_dropout_result['image']

        # Apply normalization and convert to tensor
        final_result = normalization(image=image)
        image = final_result['image']

        return image, label

test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2470, 0.2435, 0.2616,))
                                       ])

# Create Train and test dataset
train = CIFAR10WithCustomDropout('./data', train=True, coarse_dropout_prob=0.1)
test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))



