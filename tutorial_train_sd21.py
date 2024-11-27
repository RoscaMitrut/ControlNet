from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import TensorBoardLogger

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 16
logger_freq = 146
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
logger2 = TensorBoardLogger("tb_logs", name="my_model")

best_model_checkpoint = ModelCheckpoint(
    dirpath='./checkpoints/',
    filename='best-model-{epoch:02d}-{val_loss:.2f}',  # Include the metric in the filename
    monitor='train/loss_epoch',  # Replace with the metric you want to monitor
    mode='min',  # 'min' for loss, 'max' for accuracy, etc.
    save_top_k=1,  # Save only the best model
    save_weights_only=True
)

# Save the last model
last_model_checkpoint = ModelCheckpoint(
    dirpath='./checkpoints/',
    filename='last-model',
    save_top_k=1,  # Always keep only the last model
    save_last=True,  # Ensure the last model is saved
    save_weights_only=True
)

trainer = pl.Trainer(gpus=1, precision=32, logger=logger2, callbacks=[logger,best_model_checkpoint,last_model_checkpoint], max_epochs=10)

# Train!
trainer.fit(model, dataloader)
