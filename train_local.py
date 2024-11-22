# --------------------------------------------------------
# main script to train a local model
# Written by Sina Gholami
# --------------------------------------------------------
import glob
import os
import torch
from dataset.datamodule import CustomDatamodule
from models.classification import MLP, ClassificationNet
from util.apply_transformation import get_train_transformation, get_test_transformation
from util.get_models import get_baseline_model
from util.log_handler import log_results
from util.seed import set_seed
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    set_seed(42)
    comment = "local_1"
    model_architecture = "resnet50"
    pretrained = True
    resume = False
    batch_size = 10
    img_size = 224
    epochs = 10
    feature_dim = 1000

    device = "gpu" if torch.cuda.is_available() else "cpu"

    config = {"batch_size": batch_size, "epochs": epochs, "current_round": 1}
    classes = {"0": 0, "1": 1}
    data_module = CustomDatamodule(root_dir="./data", batch_size=batch_size,
                                   train_transform=get_train_transformation(img_size=img_size),
                                   test_transform=get_test_transformation(img_size=img_size),
                                   dataset_name="custom",
                                   classes=classes)
    data_module.prepare_data()
    data_module.setup("train")
    data_module.setup("val")
    data_module.setup("test")

    save_path = os.path.join(f"./checkpoints", comment, data_module.dataset_name)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False,
                                   mode="min")
    tb_logger = TensorBoardLogger(save_dir=save_path, name="tb")
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 filename=model_architecture + "_-{epoch}-{val_auc:.4f}",
                                 mode="min", monitor="val_loss", save_top_k=1)

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=config["epochs"],
        callbacks=[early_stopping, checkpoint],
        logger=[tb_logger],
        log_every_n_steps=1
    )

    encoder = get_baseline_model(pretrained=pretrained, model_architecture=model_architecture)
    model = ClassificationNet(feature_dim=feature_dim, encoder=encoder, lr=3e-5, wd=1e-6, classes=data_module.classes)
    model_path = glob.glob(os.path.join(save_path, f"{model_architecture}_*.ckpt"))
    if len(model_path) > 0:
        model = ClassificationNet.load_from_checkpoint(model_path[0], feature_dim=1000,
                                                       classes=data_module.classes, lr=3e-5, wd=1e-6,
                                                       encoder=encoder)
    else:
        model = ClassificationNet(feature_dim=feature_dim, classes=data_module.classes, lr=3e-5, wd=1e-6, encoder=encoder)
        # train the model
        trainer.fit(model=model, datamodule=data_module)
    model_path = glob.glob(os.path.join(save_path, f"{model_architecture}_*.ckpt"))
    model = ClassificationNet.load_from_checkpoint(model_path[0], feature_dim=1000,
                                                   classes=data_module.classes, lr=3e-5, wd=1e-6,
                                                   encoder=encoder)
    # Test the model
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=config["epochs"],
    )
    test_results = trainer.test(model, data_module.test_dataloader())
    config['test_set'] = data_module.dataset_name
    log_results(log_root_dir="./logs",
                classes=data_module.classes,
                results=test_results,
                comment=comment,
                dataset_name=data_module.dataset_name)

