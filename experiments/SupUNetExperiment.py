"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time
import json 
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.stats import Dice3d, Jaccard3d, Sensitivity, Specificity
from models.RecursiveUNet import RUNet
from models.UNet import UNET
from models.SupUNet import SupUNet
from experiments.InferenceAgent import InferenceAgent


class SupUNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:
        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name
        with open(config.model_config) as f:
            self.arch_config = json.load(f)
        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        self.test_data = dataset[split["test"]]

        # Do we have CUDA available?
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.model = SupUNet.build_from_config(self.arch_config)
        self.model =UNET(nclass=3,aux=False, in_channels=1)
        self.model.to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up Tensorboard. By default it saves data into runs folder. You need to launch
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data = batch['image'].to(self.device, dtype=torch.float)
            print(data.shape)
            target = batch['seg'].to(self.device)

            prediction = self.model(data)

            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])

            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)

            print(".", end='')

        print("\nTraining complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        print(f"Validating epoch {self.epoch}...")
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                
                data = batch['image'].to(self.device, dtype=torch.float)
                target = batch['seg'].to(self.device)

                prediction = self.model(data)

                prediction_softmax = F.softmax(prediction, dim=1)
                loss = self.loss_function(prediction, target[:, 0, :, :])

                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100)
        print(f"Validation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the 
        # clinical environment. 

        inference_agent = InferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []
        sensitivity_list = []
        specificity_list = []
        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            sensitivity = Sensitivity(x['seg'], pred_label)
            specificity = Specificity(x['seg'], pred_label)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            
            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc,
                "sensitivity": sensitivity,
                "specificity": specificity
                })
            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list),
            "mean_sensitivity": np.mean(sensitivity_list),
            "mean_specificity": np.mean(specificity_list)}

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
