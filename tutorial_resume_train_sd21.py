from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset_1,MyDataset_3,MyDataset_4
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
import os


assert len(sys.argv) == 4, 'Args are wrong. There should be 3 args: input_channels, batch size, epochs.'

input_channels = sys.argv[1]
assert input_channels in ['1', '3', '4'], 'Input channels must be 1, 3 or 4.'

batch_size = sys.argv[2]
batch_size = int(batch_size)
assert batch_size > 0, 'Batch size must be positive.'

epochs = sys.argv[3]
epochs = int(epochs)
assert epochs > 0, 'Epochs must be positive.'

logger_freq = int(1000/batch_size)
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

model = create_model(f'./ControlNet/models/cldm_v21_{input_channels}.yaml').cpu()
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

if input_channels == "1":
    dataset = MyDataset_1()
elif input_channels == "3":
    dataset = MyDataset_3()
elif input_channels == "4":
    dataset = MyDataset_4()

dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

logger2 = TensorBoardLogger(f"tb_logs_{input_channels}", name="my_model")

last_model_checkpoint = ModelCheckpoint(
    dirpath=f'./checkpoints/',
    save_weights_only=True,
    save_top_k=1,
    monitor="global_step",
    mode="max",
    )

trainer = pl.Trainer(gpus=1, precision=32, logger=logger2, callbacks=[logger,last_model_checkpoint], max_epochs=epochs, resume_from_checkpoint=f"./ControlNet/models/SDAug{input_channels}_19999.ckpt")

trainer.fit(model, dataloader)