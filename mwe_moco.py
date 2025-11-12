from collections import OrderedDict
from pathlib import Path
import random
from tqdm import tqdm
from typing import Union, Optional, Callable, List, Type
from PIL import ImageFilter
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoTransform:
    """Take two random transforms of one image as the query and key."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return [q, k]


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)

        return img, target

    def __getitems__(self, indices):
        samples = [self.__getitem__(idx) for idx in indices]

        return samples


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        num_blocks: list[int],
        num_classes: int = 10,
        norm_layer = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        planes = self.inplanes
        self.layer1 = self._make_layer(block, planes, num_blocks[0])
        planes *= 2
        self.layer2 = self._make_layer(block, planes, num_blocks[1], stride=2)
        planes *= 2
        self.layer3 = self._make_layer(block, planes, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(planes * block.expansion, num_classes)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x, shape=[B, 3, H , W]
        # nn.Conv2d(3, self.inplanes=16, kernel_size=3, stride=1, padding=1, bias=False)
        x = self.conv1(x)

        # norm_layer=nn.BatchNorm2d(self.inplanes=16)
        x = self.bn1(x)
        # nn.ReLU(inplace=True)
        x = self.relu(x)

        # self._make_layer(block, planes, num_blocks[0])
        x = self.layer1(x)
        # self._make_layer(block=BasicBlock, planes * 2, num_blocks[1], stride=2)
        x = self.layer2(x)
        # self._make_layer(block=BasicBlock, planes * 4, num_blocks[2], stride=2)
        x = self.layer3(x)

        # nn.AdaptiveAvgPool2d((1, 1))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x, shape=[B, num_classes]
        x = self.linear(x)

        return x


def create_resnet_model(num_classes):
    resnet_model = ResNet(BasicBlock,
                          [2, 2, 2, 2],
                          num_classes=num_classes,
                          norm_layer=nn.BatchNorm2d,
                          # norm_layer=nn.InstanceNorm2d
                          )

    return resnet_model

class MWEMoco(nn.Module):
    def __init__(self,
                 encoder,
                 feature_dim: int = 10,
                 queue_size: int = 65536,
                 momentum: float = 0.999,
                 temperature: float = 0.07
                 ):
        """
        Arguments:
            encoder: encoder network to read image and output feature
            feature_dim: feature dimension
            queue_size: number of negative keys
            momentum: moco momentum to update key encoder
            temperature: softmax temperature
        """
        super(MWEMoco, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder(num_classes=feature_dim)
        self.encoder_k = encoder(num_classes=feature_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue, shape=[C, queue_size]
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        # queue head to add new keys
        self.register_buffer("queue_head_idx", torch.zeros(1, dtype=torch.long))

    def forward(self, im_q, im_k):
        """
        Arguments:
            im_q: query images, shape=[B, C, H, W]
            im_k: key images, shape=[B, C, H, W]
        Output:
            logits, shape=[B, 1 + queue_size]
            labels, shape=[B]
        """
        # compute query features, shape=[B, C]
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # compute positive key features, shape=[B, C]
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k, idx = self.batch_shuffle(im_k)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self.batch_unshuffle(k, idx)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits, shape=[B, 1]
        pos_logits = torch.einsum("bc,bc->b", [q, k]).unsqueeze(-1)
        # negative logits, shape=[B, queue_size]
        neg_logits = torch.einsum("bc,cq->bq", [q, self.queue.clone().detach()])

        # logits, shape=[B, 1 + queue_size]
        logits = torch.cat([pos_logits, neg_logits], dim=1)

        # apply temperature, shape=[B, 1 + queue_size]
        logits /= self.temperature

        # labels: positive key indicators, shape=[B]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue positive key features, shape=[B, C]
        self._dequeue_and_enqueue(k)

        # logits, shape=[B, 1 + queue_size]
        # labels, shape=[B]
        return logits, labels

    @torch.no_grad()
    def batch_shuffle(self, x):
        idx = torch.randperm(x.shape[0]).to(x.device)
        return x[idx], idx

    @torch.no_grad()
    def batch_unshuffle(self, x, idx):
        inv_idx = torch.argsort(idx)
        return x[inv_idx]

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = (param_k.data * self.momentum +
                            param_q.data * (1.0 - self.momentum))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        """
        Arguments:
            keys: key features, shape=[B, C]
        """
        batch_size = keys.shape[0]

        queue_head_idx = int(self.queue_head_idx)
        # for simplicity
        assert self.queue_size % batch_size == 0, f"queue_size={self.queue_size} batch_size={batch_size}"

        # replace the keys at queue_head_idx (dequeue and enqueue)
        self.queue[:, queue_head_idx: queue_head_idx + batch_size] = keys.T
        queue_head_idx = (queue_head_idx + batch_size) % self.queue_size  # move pointer

        self.queue_head_idx[0] = queue_head_idx


@torch.no_grad()
def topk_correct_counts(output, target, topk=(1, 5)):
    """
    Computes the number of correct predictions for each top-k and batch size.

    Args:
        output: logits or probabilities, shape [batch_size, num_classes]
        target: ground-truth labels, shape [batch_size]
        topk: tuple of k values, e.g. (1, 5)

    Returns:
        correct_counts: list of number of correct predictions per k
    """
    maxk = max(topk)

    # pred, shape=[B, maxk]
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    # pred, shape: [maxk, B]
    pred = pred.t()
    # correct, shape: [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k, shape: [1]
        correct_k = correct[:k].reshape(-1).float().sum(dim=0)
        res.append(int(correct_k))
    return res


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n: int = 1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def update_metrics(mode,
                   epoch,
                   epochs,
                   step,
                   steps,
                   outputs,
                   labels,
                   loss,
                   meters,
                   pbar=None):
    correct1, correct5 = topk_correct_counts(outputs,
                                             labels,
                                             topk=(1, 5))
    batch_size = labels.shape[0]

    losses = meters["Loss"]
    top1 = meters["Acc@1"]
    top5 = meters["Acc@5"]
    losses.update(loss.item())
    top1.update(correct1, batch_size)
    top5.update(correct5, batch_size)

    # pbar.set_description(f"[{epoch + 1}, {step + 1:05d}]")
    metrics = OrderedDict([
            (f"{mode} Epoch", f"{epoch + 1:02d}/{epochs:02d}"),
            ("Step", f"{step + 1:05d}/{steps:05d}"),
            ("Loss", f"{losses.avg:.3f}"),
            ("Acc@1", f"{top1.avg:.2%}"),
            ("Acc@5", f"{top5.avg:.2%}"),
        ])

    if pbar:
        pbar.set_postfix(metrics)


def train_one_epoch(epoch,
                    epochs,
                    model,
                    train_dataloader,
                    optimizer,
                    criterion,
                    pbar,
                    model_eval=False):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if model_eval:
        model.eval()
    else:
        model.train()

    meters = OrderedDict([
        ("Loss", AverageMeter()),
        ("Acc@1", AverageMeter()),
        ("Acc@5", AverageMeter()),
    ])

    for step, data in enumerate(train_dataloader):
        # labels, shape=[B]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # logits, shape=[B, num_classes]
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        update_metrics("Train",
                       epoch,
                       epochs,
                       step,
                       len(train_dataloader),
                       outputs,
                       labels,
                       loss,
                       meters,
                       pbar)

    return meters


def pretrain_one_epoch(epoch,
                       epochs,
                       model,
                       train_dataloader,
                       optimizer,
                       criterion,
                       pbar):
    model.train()
    meters = OrderedDict([
        ("Loss", AverageMeter()),
        ("Acc@1", AverageMeter()),
        ("Acc@5", AverageMeter()),
    ])

    for step, data in enumerate(train_dataloader):
        inputs, _ = data
        im_q, img_k = inputs
        im_q = im_q.to(device)
        img_k = img_k.to(device)

        optimizer.zero_grad()

        # logits, shape=[B, 1 + queue_size]
        # labels, shape=[B]
        outputs, labels = model(im_q=im_q, im_k=img_k)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        update_metrics("Pretrain",
                       epoch,
                       epochs,
                       step,
                       len(train_dataloader),
                       outputs,
                       labels,
                       loss,
                       meters,
                       pbar)

    return meters


@torch.no_grad()
def test_one_epoch(epoch,
                   epochs,
                   model,
                   test_dataloader,
                   criterion,
                   pbar):
    # model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
    # model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    meters = OrderedDict([
        ("Loss", AverageMeter()),
        ("Acc@1", AverageMeter()),
        ("Acc@5", AverageMeter()),
    ])

    for step, data in enumerate(test_dataloader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # outputs, shape=[B, num_classes]
        outputs = model(images)

        loss = criterion(outputs, labels)
        update_metrics("Test",
                       epoch,
                       epochs,
                       step,
                       len(test_dataloader),
                       outputs,
                       labels,
                       loss,
                       meters,
                       pbar)

    return meters

print("Completed classes and methods definiton")

seed = 17
set_seed(seed)

data_root_path = "./data"
# batch_size = 4
batch_size = 128
queue_size = batch_size * 100
epochs = 10
pretrain_epochs = 10
finetune_epochs = epochs - pretrain_epochs


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            normalize,
        ]
    )

# MoCo v2's aug
pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply(
        [GaussianBlur([0.1, 2.0])],
        p=0.5,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

origin_train_dataset = torchvision.datasets.CIFAR10(root=data_root_path,
                                                    train=True,
                                                    download=True)
train_dataset = torchvision.datasets.CIFAR10(root=data_root_path,
                                             train=True,
                                             download=True,
                                             transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

targets = origin_train_dataset.targets
# Define stratified splitter
splitter = StratifiedShuffleSplit(
    n_splits=1,     # number of re-shuffles
    test_size=0.1,
    random_state=17
)

# Get train and subset indices
for pretrain_idx, finetune_idx in splitter.split(np.zeros(len(targets)), targets):
    print(len(pretrain_idx), len(finetune_idx))
    # pretrain_dataset = TransformSubset(origin_train_dataset,
    #                                    pretrain_idx,
    #                                    transform=TwoTransform(pretrain_transform))
    finetune_dataset = TransformSubset(origin_train_dataset,
                                       finetune_idx,
                                       transform=transform)

pretrain_dataset = torchvision.datasets.CIFAR10(root=data_root_path,
                                                train=True,
                                                download=True,
                                                transform=TwoTransform(pretrain_transform))
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2,
                                                  drop_last=True)

# finetune_dataset = torchvision.datasets.CIFAR10(root=data_root_path,
#                                                 train=True,
#                                                 download=True,
#                                                 transform=transform)
finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root=data_root_path,
                                       train=False,
                                       download=True,
                                       transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

class_names = test_dataset.classes
num_classes = len(class_names)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Train
set_seed(seed)
# model = create_resnet_model(num_classes).to(device)
model = resnet50(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

pbar = tqdm(range(epochs))

for epoch in pbar:
    train_one_epoch(epoch,
                    epochs,
                    model,
                    finetune_dataloader,
                    optimizer,
                    criterion,
                    pbar)

    tqdm.write("")
    # model_path = "./cifar_model.pth"
    # torch.save(model.state_dict(), model_path)
    test_meters = test_one_epoch(epoch,
                                 epochs,
                                 model,
                                 test_dataloader,
                                 criterion,
                                 pbar)

    tqdm.write("")

# Pretrain
set_seed(seed)
# moco_model = MWEMoco(create_resnet_model,
#                      feature_dim=num_classes,
#                      queue_size=queue_size).to(device)
moco_model = MWEMoco(resnet50,
                     feature_dim=num_classes,
                     queue_size=queue_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(moco_model.parameters(), lr=0.1, momentum=0.9)

pbar = tqdm(range(pretrain_epochs))

for epoch in pbar:
    pretrain_one_epoch(epoch,
                       pretrain_epochs,
                       moco_model,
                       pretrain_dataloader,
                       optimizer,
                       criterion,
                       pbar)

    tqdm.write("")

# Finetune

freeze_layers = False
set_seed(seed)
# model = create_resnet_model(num_classes).to(device)
model = resnet50(num_classes=num_classes).to(device)

if isinstance(model, ResNet):
    linear_weight_names = ["fc.weight", "fc.bias"]
else:
    linear_weight_names = ["linear.weight", "linear.bias"]

if freeze_layers:
    # freeze all layers but the last linear
    for name, param in model.named_parameters():
        if name not in linear_weight_names:
            param.requires_grad = False

# init the linear layer

if isinstance(model, ResNet):
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
else:
    model.linear.weight.data.normal_(mean=0.0, std=0.01)
    model.linear.bias.data.zero_()

state_dict = {key[len("encoder_q."):]: value
        for key, value in moco_model.state_dict().items()
        if (key.startswith("encoder_q")
            and not key.startswith("encoder_q.linear")
            )
    }

msg = model.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == set(linear_weight_names)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

pbar = tqdm(range(epochs))

for epoch in pbar:
    train_one_epoch(epoch,
                    epochs,
                    model,
                    finetune_dataloader,
                    optimizer,
                    criterion,
                    pbar,
                    model_eval=freeze_layers)

    tqdm.write("")
    # model_path = "./cifar_model.pth"
    # torch.save(model.state_dict(), model_path)
    test_meters = test_one_epoch(epoch,
                                 epochs,
                                 model,
                                 test_dataloader,
                                 criterion,
                                 pbar)

    tqdm.write("")


print("Finished!")


acc_infos_list = []

pretrain_epochs_range = range(1, 10 + 1)
finetune_epochs = 10

for pretrain_epochs in pretrain_epochs_range:
    print("=" * 80)
    print(f"pretrain_epochs={pretrain_epochs} / {len(pretrain_epochs_range)}")
    print("-" * 80)
    moco_model = MWEMoco(create_resnet_model,
                         feature_dim=num_classes,
                         queue_size=batch_size * 100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(moco_model.parameters(), lr=0.1, momentum=0.9)

    pbar = tqdm(range(pretrain_epochs))

    for epoch in pbar:
        pretrain_one_epoch(epoch,
                           pretrain_epochs,
                           moco_model,
                           pretrain_dataloader,
                           optimizer,
                           criterion,
                           pbar)

    model = create_resnet_model(num_classes).to(device)

    state_dict = {
        key[len("encoder_q."):]: value
        for key, value in moco_model.state_dict().items()
        if (key.startswith("encoder_q")
            and not key.startswith("encoder_q.linear"))
    }

    model.load_state_dict(state_dict, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    acc_infos = []
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        train_one_epoch(epoch,
                        epochs,
                        model,
                        train_dataloader,
                        optimizer,
                        criterion,
                        pbar)

        tqdm.write("")
        # model_path = "./cifar_model.pth"
        # torch.save(model.state_dict(), model_path)
        test_meters = test_one_epoch(epoch,
                                     epochs,
                                     model,
                                     test_dataloader,
                                     criterion,
                                     pbar)

        tqdm.write("")

        accuracy = test_meters["Acc@1"].avg
        acc_info = dict(epoch=epoch, accuracy=accuracy)

        acc_infos.append(acc_info)

    finetune_acc_infos = dict(pretrain_epochs=pretrain_epochs, acc_infos=acc_infos)
    acc_infos_list.append(finetune_acc_infos)
    print("=" * 80)

print("Finished!")

