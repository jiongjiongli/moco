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
from torchvision.models import resnet50
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from mwe_moco import (set_seed,
                      create_test_data,
                      run_pretrain,
                      launch_train,
                      launch_finetune)


def launch_pretrain(data_root_path,
                    batch_size,
                    num_workers,
                    num_classes,
                    model_type,
                    queue_size,
                    epochs,
                    learning_rate,
                    optim_momentum,
                    save_every,
                    moco_snapshot_path):
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")

    if world_size > 1:
        mp.spawn(run_pretrain,
                 args=(world_size,
                       seed,
                       data_root_path,
                       batch_size,
                       num_workers,
                       num_classes,
                       model_type,
                       queue_size,
                       epochs,
                       learning_rate,
                       optim_momentum,
                       save_every,
                       moco_snapshot_path),
                 nprocs=world_size)
    else:
        run_pretrain(0,
                     world_size,
                     seed,
                     data_root_path,
                     batch_size,
                     num_workers,
                     num_classes,
                     model_type,
                     queue_size,
                     epochs,
                     learning_rate,
                     optim_momentum,
                     save_every,
                     moco_snapshot_path)



seed = 17
set_seed(seed)

data_root_path = "./data"
# batch_size = 4
batch_size = 128
num_workers = 2
queue_size = batch_size * 100
train_epochs = 5
pretrain_epochs = 5
finetune_epochs = 5

world_size = torch.cuda.device_count()
multigpu = world_size > 1

test_data = create_test_data(data_root_path,
                                   batch_size,
                                   num_workers)
test_dataset = test_data["dataset"]
test_dataloader = test_data["dataloader"]

class_names = test_dataset.classes
num_classes = len(class_names)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_type = "resnet50"

# Train
train_learning_rate = 0.01
train_optim_momentum = 0.9
train_snapshot_path = "train_weights.pth"

launch_train(seed,
             device,
             data_root_path,
             batch_size,
             num_workers,
             num_classes,
             model_type,
             train_epochs,
             train_learning_rate,
             train_optim_momentum,
             train_snapshot_path
             )


# Pretrain
pretrain_learning_rate = 0.01
pretrain_optim_momentum = 0.9
save_every = 1
moco_snapshot_path = "moco_weights.pth"

launch_pretrain(data_root_path,
                batch_size,
                num_workers,
                num_classes,
                model_type,
                queue_size,
                pretrain_epochs,
                pretrain_learning_rate,
                pretrain_optim_momentum,
                save_every,
                moco_snapshot_path)

# Finetune
finetune_learning_rate = 0.01
finetune_optim_momentum = 0.9
finetune_snapshot_path = "finetune_weights.pth"

launch_finetune(seed,
                device,
                data_root_path,
                batch_size,
                num_workers,
                num_classes,
                model_type,
                finetune_epochs,
                finetune_learning_rate,
                finetune_optim_momentum,
                finetune_snapshot_path,
                moco_snapshot_path
             )

print("Finished!")


# acc_infos_list = []

# pretrain_epochs_range = range(1, 20 + 1)
# finetune_epochs = 10

# for pretrain_epochs in pretrain_epochs_range:
#     print("=" * 80)
#     print(f"pretrain_epochs={pretrain_epochs} / {len(pretrain_epochs_range)}")
#     print("-" * 80)
#     set_seed(seed)
#     # moco_model = MWEMoco(create_resnet_model,
#     #                      feature_dim=num_classes,
#     #                      queue_size=batch_size * 100).to(device)
#     moco_model = MWEMoco(resnet50,
#                          feature_dim=num_classes,
#                          queue_size=queue_size).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(moco_model.parameters(), lr=0.1, momentum=0.9)

#     pbar = tqdm(range(pretrain_epochs))

#     for epoch in pbar:
#         pretrain_one_epoch(epoch,
#                            pretrain_epochs,
#                            moco_model,
#                            pretrain_dataloader,
#                            optimizer,
#                            criterion,
#                            pbar)

#     set_seed(seed)
#     model = create_encoder(moco_model, num_classes, freeze_layers)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#     acc_infos = []
#     pbar = tqdm(range(epochs))

#     for epoch in pbar:
#         train_one_epoch(epoch,
#                         epochs,
#                         model,
#                         train_dataloader,
#                         optimizer,
#                         criterion,
#                         pbar)

#         tqdm.write("")
#         # model_path = "./cifar_model.pth"
#         # torch.save(model.state_dict(), model_path)
#         test_meters = test_one_epoch(epoch,
#                                      epochs,
#                                      model,
#                                      test_dataloader,
#                                      criterion,
#                                      pbar)

#         tqdm.write("")

#         accuracy = test_meters["Acc@1"].avg
#         acc_info = dict(epoch=epoch, accuracy=accuracy)

#         acc_infos.append(acc_info)

#     finetune_acc_infos = dict(pretrain_epochs=pretrain_epochs, acc_infos=acc_infos)
#     acc_infos_list.append(finetune_acc_infos)
#     print("=" * 80)

# print("Finished!")
