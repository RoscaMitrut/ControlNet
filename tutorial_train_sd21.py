from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tutorial_dataset import MyDataset_1,MyDataset_3,MyDataset_4
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import TensorBoardLogger

import sys

assert len(sys.argv) == 2, 'Args are wrong. There should be 1 arg: input_channels.'

input_channels = sys.argv[1]
assert input_channels in ['1', '3', '4'], 'Input channels must be 1, 3 or 4.'


# Configs
resume_path = './ControlNet/models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 125
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(f'./ControlNet/models/cldm_v21_{input_channels}.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
if input_channels == "1":
    dataset = MyDataset_1()
elif input_channels == "3":
    dataset = MyDataset_3()
elif input_channels == "4":
    dataset = MyDataset_4()

dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
logger2 = TensorBoardLogger(f"tb_logs_{input_channels}", name="my_model")

# Save the last model
last_model_checkpoint = ModelCheckpoint(
    dirpath=f'./checkpoints_{input_channels}/',
    filename='last-model',
    save_top_k=1,  # Always keep only the last model
    save_last=True,  # Ensure the last model is saved
    save_weights_only=True
)

trainer = pl.Trainer(gpus=1, precision=32, logger=logger2, callbacks=[logger,last_model_checkpoint], max_epochs=1)

# Train!
trainer.fit(model, dataloader)
