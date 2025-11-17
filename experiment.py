from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.models import resnet50

from PIL import ImageFilter  # remove if not used
# If ImageFilter is unused in mwe_moco, delete this line.

from mwe_moco import (
    set_seed,
    DataManager,
    run_pretrain,
    launch_train,
    launch_finetune
)


def launch_pretrain(config,
                    device_type,
                    num_classes):
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    multigpu = device_type == "gpu" and world_size > 1
    rank = 0 if multigpu else -1

    if multigpu:
        mp.spawn(run_pretrain,
                 args=(world_size,
                       config,
                       device_type,
                       num_classes),
                 nprocs=world_size)
    else:
        run_pretrain(rank,
                     world_size,
                     config,
                     device_type,
                     num_classes)


# Config
# batch_size = 4
batch_size = 128
config_dict = dict(
    seed = 17,

    # Data
    data_root_path = "./data",
    batch_size = batch_size,
    num_workers = 2,

    # Pretrain
    queue_size = batch_size * 100,
    pretrain_epochs = 5,
    # device_type = "gpu",
    pretrain_learning_rate = 0.01,
    pretrain_optim_momentum = 0.9,
    save_every = 1,
    pretrain_snapshot_path = "moco_weights.pth",

    # Train
    train_epochs = 5,
    train_learning_rate = 0.01,
    train_optim_momentum = 0.9,
    train_snapshot_path = "train_weights.pth",
    train_freeze_layers = False,
    train_multigpu = False,

    # Finetune
    finetune_epochs = 5,
    finetune_learning_rate = 0.01,
    finetune_optim_momentum = 0.9,
    finetune_snapshot_path = "finetune_weights.pth",
    finetune_freeze_layers = False,
    finetune_multigpu = False,

    # Model
    model_type = "resnet50",
)

config = SimpleNamespace(**config_dict)

set_seed(config.seed)

test_data = DataManager.create_test_data(config)
test_dataset = test_data["dataset"]
test_dataloader = test_data["dataloader"]

class_names = test_dataset.classes
num_classes = len(class_names)


# Train
device_type = "gpu"
launch_train(config, device_type, num_classes)

# Pretrain
device_type = "gpu"
launch_pretrain(config,
                device_type,
                num_classes)

# Finetune
device_type = "gpu"
launch_finetune(config, device_type, num_classes)

print("Finished!")


# Pretrain
device_type = "cpu"
config.batch_size = 1
config.queue_size = batch_size * 1
launch_pretrain(config,
                device_type,
                num_classes)

# Finetune
device_type = "cpu"
launch_finetune(config, device_type, num_classes)

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
