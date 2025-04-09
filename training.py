##General imports
import csv
import os
import time
from datetime import datetime
import shutil
import copy
import numpy as np
from functools import partial
import platform
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
##Torch imports
import torch.nn.functional as F
import torch
from cgcnn import CGCNN
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import DataParallel
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp


#sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge,Lasso, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error,roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

##Matdeeplearn imports

import cgcnn
import process
import training


################################################################################
#  Training functions
################################################################################

##Train step, runs model in train mode
def train(model, optimizer, loader, loss_method, rank):
    model.train()
    loss_all = 0
    count = 0
    criterion = torch.nn.SmoothL1Loss()
    #criterion = torch.nn.MSELoss()
    for data in loader:
        data = data.to(rank)
        optimizer.zero_grad()
        output,_ = model(data)
        loss = getattr(F, loss_method)(output, data.y)
        output_size = output.size(0)
        loss_all += loss.detach() * output_size

        loss.backward()
        # clip = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        count = count + output_size

    loss_all = loss_all / count
    return loss_all


##Evaluation step, runs model in eval mode
def evaluate(loader, model, loss_method, rank, out=False):
    model.eval()
    loss_all = 0
    count = 0
    for data in loader:
        data = data.to(rank)
        with torch.no_grad():
            output,_ = model(data)
            loss = getattr(F, loss_method)(output, data.y)
            output_size = output.size(0)
            loss_all += loss.detach() * output_size
            # output = model(data)
            # loss = getattr(F, loss_method)(output, data.y)
            # loss_all += loss * output.size(0)
            if out == True:
                if count == 0:
                    ids = [item for sublist in data.structure_id for item in sublist]
                    ids = [item for sublist in ids for item in sublist]
                    predict = output.data.cpu().numpy()
                    target = data.y.cpu().numpy()
                else:
                    ids_temp = [
                        item for sublist in data.structure_id for item in sublist
                    ]
                    ids_temp = [item for sublist in ids_temp for item in sublist]
                    ids = ids + ids_temp
                    predict = np.concatenate(
                        (predict, output.data.cpu().numpy()), axis=0
                    )
                    target = np.concatenate((target, data.y.cpu().numpy()), axis=0)
            count = count + output_size

    loss_all = loss_all / count

    if out == True:
        test_out = np.column_stack((ids, target, predict))
        return loss_all, test_out
    elif out == False:
        return loss_all


##Model trainer
def trainer(
    rank,
    world_size,
    model,
    optimizer,
    scheduler,
    loss,
    train_loader,
    val_loader,
    train_sampler,
    epochs,
    verbosity,
    filename = "my_model_temp.pth",
):
    torch.cuda.empty_cache()
    train_error = val_error = test_error = epoch_time = float("NaN")
    train_start = time.time()
    best_val_error = 1e10
    model_best = model
    print("torch memory", torch.cuda.memory_summary())
    ##Start training over epochs loop
    for epoch in range(1, epochs + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        if rank not in ("cpu", "cuda"):
            train_sampler.set_epoch(epoch)
        ##Train model
        print("Training Epoch", epoch)
        train_error = train(model, optimizer, train_loader, loss, rank=rank)
        if rank not in ("cpu", "cuda"):
            torch.distributed.reduce(train_error, dst=0)
            train_error = train_error / world_size

        ##Get validation performance
        if rank not in ("cpu", "cuda"):
            dist.barrier()
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                val_error = evaluate(
                    val_loader, model.module, loss, rank=rank, out=False
                )
            else:
                val_error = evaluate(val_loader, model, loss, rank=rank, out=False)

        ##Train loop timings
        epoch_time = time.time() - train_start
        train_start = time.time()

        ##remember the best val error and save model and checkpoint        
        if val_loader != None and rank in (0, "cpu", "cuda"):
            if val_error == float("NaN") or val_error < best_val_error:
                if rank not in ("cpu", "cuda"):
                    model_best = copy.deepcopy(model.module)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
                else:
                    model_best = copy.deepcopy(model)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "full_model": model,
                        },
                        filename,
                    )
            best_val_error = min(val_error, best_val_error)
        elif val_loader == None and rank in (0, "cpu", "cuda"):
            if rank not in ("cpu", "cuda"):
                model_best = copy.deepcopy(model.module)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )
            else:
                model_best = copy.deepcopy(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    filename,
                )

        ##scheduler on train error
        scheduler.step(train_error)

        ##Print performance
        if epoch % verbosity == 0:
            if rank in (0, "cpu", "cuda"):
                print(
                    "Epoch: {:04d}, Learning Rate: {:.6f}, Training Error: {:.5f}, Val Error: {:.5f}, Time per epoch (s): {:.5f}".format(
                        epoch, lr, train_error, val_error, epoch_time
                    )
                )

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    return model_best


##Write results to csv file
def write_results(output, filename):
    shape = output.shape
    with open(filename, "w") as f:
        csvwriter = csv.writer(f)
        for i in range(0, len(output)):
            if i == 0:
                csvwriter.writerow(
                    ["ids"]
                    + ["target"] * int((shape[1] - 1) / 2)
                    + ["prediction"] * int((shape[1] - 1) / 2)
                )
            elif i > 0:
                csvwriter.writerow(output[i - 1, :])


##Pytorch ddp setup
def ddp_setup(rank, world_size):
    if rank in ("cpu", "cuda"):
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if platform.system() == 'Windows':
        dist.init_process_group("gloo", rank=rank, world_size=world_size)    
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True


##Pytorch model setup
def model_setup(
    rank,
    model_name,
    model_params,
    dataset,
    load_model=False,
    path=None,
    print_model=True,
):

    model = CGCNN(
        data=dataset,
        dim1=model_params.get("dim1", 100),
        dim2=model_params.get("dim2", 150),
        pre_fc_count=model_params.get("pre_fc_count", 1),
        gc_count=model_params.get("gc_count", 4),
        post_fc_count=model_params.get("post_fc_count", 3),
        pool=model_params.get("pool", "global_mean_pool"),
        pool_order=model_params.get("pool_order", "early"),
        batch_norm=model_params.get("batch_norm", "True"),
        batch_track_stats=model_params.get("batch_track_stats", "True"),
        act=model_params.get("act", "relu"),
        dropout_rate=model_params.get("dropout_rate", 0.0)
    ).to(rank)


    if load_model:
        model_path = os.path.join(os.getcwd(), path)
        assert os.path.exists(model_path), "Saved model not found"
        if str(rank) in ("cpu"):
            saved= torch.load(model_path, map_location=torch.device("cpu"))
        else:
            saved = torch.load(model_path)
        # print("dict", saved.keys())
        # print("input_encoder.weight",saved["input_encoder.weight"].shape)
        # print("edge_encoder.weight", saved["edge_encoder.weight"].shape)
        # print("gnns.0.convs.0.nn.layers.0.weight", saved["gnns.0.convs.0.nn.layers.0.weight"].shape)
        # print("gnns.0.convs.0.nn.layers.1.weight", saved["gnns.0.convs.0.nn.layers.1.weight"].shape)
        model.load_state_dict(saved["model_state_dict"])
        # optimizer.load_state_dict(saved['optimizer_state_dict'])
    # DDP
    if rank not in ("cpu", "cuda"):
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
        if print_model == True and rank in (0, "cpu", "cuda"):
          print(model)  
            
    return model


##Pytorch loader setup
def loader_setup(
    train_ratio,
    val_ratio,
    test_ratio,
    batch_size,
    dataset,
    rank,
    seed,
    world_size=0,
    num_workers=0,
):
    ##Split datasets
    train_dataset, val_dataset, test_dataset = process.split_data(
        dataset, train_ratio, val_ratio, test_ratio, seed
    )

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    ##Load data
    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    # may scale down batch size if memory is an issue
    if rank in (0, "cpu", "cuda"):
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def loader_setup_CV(index, batch_size, dataset, rank, world_size=0, num_workers=0):
    ##Split datasets
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    ##DDP
    if rank not in ("cpu", "cuda"):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
    elif rank in ("cpu", "cuda"):
        train_sampler = None

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank in (0, "cpu", "cuda"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader, train_sampler, train_dataset, test_dataset


################################################################################
#  Trainers
################################################################################

###Regular training with train, val, test split
def train_regular(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size

    ##Get dataset
    dataset = process.get_dataset(data_path=data_path, target_index= training_parameters["target_index"], reprocess=True,  model_name = model_parameters["model"])

    if rank not in ("cpu", "cuda"):
        dist.barrier()

    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )
    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        job_parameters["load_model"],
        job_parameters["model_path"],
        model_parameters.get("print_model", False),
    )
    #
    # ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )
    #
    ##Start training
    model = trainer(
        rank=rank,
        world_size=world_size,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=training_parameters["loss"],
        train_loader=train_loader,
        val_loader=None,
        train_sampler=train_sampler,
        # epochs=model_parameters["epochs"],
        epochs=10,
        verbosity=training_parameters["verbosity"],
        filename="CGCNN_MP_robofail_50per_fe.pth",
    )
    #
    # model_path =os.path.join(os.getcwd(), 'Results/Models', 'CGCNN_100_epochs.pth')
    # torch.save(model.state_dict(), model_path)

    if type(model).__name__ =="GraphJepaE3":

         # cmb = Combined(model,nin=8,nout=1)
         predict_jepa(train_loader, test_loader, model, rank)
         #finetune_jepa(train_loader, test_loader, model, rank)


    if rank in (0, "cpu", "cuda"):

        train_error = val_error = test_error = float("NaN")

        ##workaround to get training output in DDP mode
        ##outputs are slightly different, could be due to dropout or batchnorm?
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=model_parameters["batch_size"],
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,
        # )



        ##Get train error in eval mode
        train_error, train_out = evaluate(
            train_loader, model, training_parameters["loss"], rank, out=True
        )
        print("Train Error: {:.5f}".format(train_error))

        ##Get val error
        if val_loader != None:
            val_error, val_out = evaluate(
                val_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Val Error: {:.5f}".format(val_error))

        ##Get test error
        if test_loader != None:
            test_error, test_out = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

        ##Save model
        if job_parameters["save_model"] == "True":

            if rank not in ("cpu", "cuda"):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "full_model": model,
                    },
                    job_parameters["model_path"],
                )

        ##Write outputs
        if job_parameters["write_output"] == "True":

            write_results(
                train_out, str(job_parameters["job_name"]) + "_train_outputs.csv"
            )
            if val_loader != None:
                write_results(
                    val_out, str(job_parameters["job_name"]) + "_val_outputs.csv"
                )
            if test_loader != None:
                write_results(
                    test_out, str(job_parameters["job_name"]) + "_test_outputs.csv"
                )

        if rank not in ("cpu", "cuda"):
            dist.destroy_process_group()

        ##Write out model performance to file
        error_values = np.array((train_error.cpu(), val_error.cpu(), test_error.cpu()))
        if job_parameters.get("write_error") == "True":
            np.savetxt(
                job_parameters["job_name"] + "_errorvalues.csv",
                error_values[np.newaxis, ...],
                delimiter=",",
            )

        return error_values


class CnnRegressor(torch.nn.Module):
    # defined the initialization method
    def __init__(self, batch_size, inputs, outputs):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()
        # store the parameters
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        # define the input layer
        self.input_layer = torch.nn.Conv2d(inputs, batch_size, 1, stride=1)

        # define max pooling layer
        self.max_pooling_layer = torch.nn.MaxPool2d(1)

        # define other convolutional layers
        self.conv_layer1 = torch.nn.Conv2d(batch_size, 128, kernel_size=1, stride=3)
        self.conv_layer2 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=3)
        self.conv_layer3 = torch.nn.Conv2d(256, 512, kernel_size=1, stride=3)
        self.conv_layer4 = torch.nn.Conv2d(512, 512, kernel_size=1, stride=3)

        # define the flatten layer
        self.flatten_layer = torch.nn.Flatten()

        # define the linear layer
        self.linear_layer = torch.nn.Linear(9728, 128)

        # define the output layer
        self.output_layer = torch.nn.Linear(128, outputs)

    # define the method to feed the inputs to the model
    def forward(self, input):
        # input is reshaped to the 1D array and fed into the input layer
        # input = input.reshape((self.batch_size, self.inputs, 1))
        # ReLU is applied on the output of input layer
        output = torch.nn.functional.relu(self.input_layer(input))

        # max pooling is applied and then Convolutions are done with ReLU
        output = self.max_pooling_layer(output)
        output = torch.nn.functional.relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = torch.nn.functional.relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = torch.nn.functional.relu(self.conv_layer3(output))

        # output = self.max_pooling_layer(output)
        # output = torch.nn.functional.relu(self.conv_layer4(output))
        # flatten layer is applied
        output = self.flatten_layer(output)

        # linear layer and ReLu is applied
        output = torch.nn.functional.relu(self.linear_layer(output))

        # finally, output layer is applied
        output = self.output_layer(output)
        return output



# def basic_nn(X_tr, y_tr, X_te, y_te, n_epochs=2, batch_size=32):_
#     X_train = torch.from_numpy(X_tr).cuda()
#     X_test = torch.from_numpy(X_te).cuda()
#     y_train = torch.from_numpy(y_tr).cuda()
#     y_test = torch.from_numpy(y_te).cuda()

#     model_ann = torch.nn.Sequential(
#         torch.nn.Linear(500,256),
#         torch.nn.ReLU(),
#         torch.nn.Linear(256, 128),
#         # torch.nn.ReLU(),
#         # torch.nn.Linear(128, 64),
#         # torch.nn.ReLU(),
#         # torch.nn.Linear(64, 32),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 1)
#     )

#     #model = CnnRegressor(128, X_train.shape[1], 1)

#     model_ann.cuda()
#     model_ann.train()
#     test_loss,train_loss = [],[]
#     loss_fn = torch.nn.MSELoss()
#     #loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model_ann.parameters(), lr=0.001)
#     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#     for epoch in range(n_epochs):
#         train_loss = []
#         for i in range(0, len(X_train), batch_size):
#             Xbatch = X_train[i:i + batch_size]
#             if Xbatch.shape[0]!=batch_size:
#                 break
#             Xbatch =Xbatch.cuda()
#             y_pred = model_ann(Xbatch)
#             ybatch = y_train[i:i + batch_size].float()
#             #ybatch = y_train[i:i + batch_size].type(torch.LongTensor).cuda()
#             y_pred= y_pred.cuda()
#             loss = loss_fn(y_pred, ybatch)
#             train_loss.append(float(loss))
#         print(f'Finished epoch {epoch}, learning rate {scheduler.get_lr()}loss {np.mean(train_loss)}')

#     model_ann.eval()
#     total, correct=0,0
#     # compute accuracy (no_grad is optional)
#     with torch.no_grad():

#             for i in range(0, len(X_train), batch_size):
#                 Xbatch_train = X_train[i:i + batch_size]
#                 if Xbatch_train.shape[0] != batch_size:
#                     break
#                 nn_predictions_train = model_ann(Xbatch_train)
#                 # output = torch.max(nn_predictions_test, 1)
#                 ybatch_train = y_train[i:i + batch_size].float()
#                 # correct += (output == ybatch_test).sum().item()
#                 # total += ybatch_test.size(0)
#                 y_true = ybatch_train.detach().cpu().numpy()
#                 error = nn_predictions_train.detach().cpu().numpy()
#                 train_error = mean_absolute_error(y_true, error)
#                 train_loss.append(float(train_error))

#             for i in range(0, len(X_test), batch_size):
#                 Xbatch = X_test[i:i + batch_size]
#                 if Xbatch.shape[0] != batch_size:
#                     break
#                 nn_predictions_test = model_ann(Xbatch)
#                 # output = torch.max(nn_predictions_test, 1)
#                 ybatch_test = y_test[i:i + batch_size].float()
#                 # correct += (output == ybatch_test).sum().item()
#                 # total += ybatch_test.size(0)
#                 y_true =ybatch_test.detach().cpu().numpy()
#                 error = nn_predictions_test.detach().cpu().numpy()
#                 test_error = mean_absolute_error(y_true,error)
#                 test_loss.append(float(test_error))

#     print(f"NN Train Error {np.mean(train_loss)} NN Test Error {np.mean(test_loss)}")
#     #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


class MyEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA.encode(x)
        x2 = self.modelB(x1)
        return x2



def finetune_jepa(train_loader, test_loader,jepa_model, rank, n_epochs=10):
    model_mlp = torch.nn.Sequential(torch.nn.Linear(500,100),torch.nn.ReLU(),torch.nn.Linear(100, 100),torch.nn.ReLU(),torch.nn.Linear(100, 1))

    model_ensemble = MyEnsemble(jepa_model,model_mlp)
    model_ensemble.to(rank)
    model_ensemble.train()
    test_error,train_error = [],[]
    loss_fn = torch.nn.MSELoss()
    #loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_ensemble.parameters(), lr=0.00005)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    ### Extracting training features and labels in Scikit-Learn form
    for epoch in range(n_epochs):
        train_loss = []
        for data in train_loader:
            data.to(rank)
            pred = model_ensemble(data)
            loss = loss_fn(pred, data.y)
            train_loss.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Finished epoch {epoch}, learning rate {scheduler.get_lr()}loss {np.mean(train_loss)}')

    model_ensemble.eval()
    # total, correct = 0, 0
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        for data in train_loader:
            data.to(rank)
            nn_predictions_test = model_ensemble(data)
            y_true = data.y.detach().cpu().numpy()
            error = nn_predictions_test.detach().cpu().numpy()
            train_error_value = mean_absolute_error(y_true, error)
            train_error.append(float(train_error_value))

        for data in test_loader:
            data.to(rank)
            nn_predictions_test = model_ensemble(data)
            y_true = data.y.detach().cpu().numpy()
            error = nn_predictions_test.detach().cpu().numpy()
            test_error_value = mean_absolute_error(y_true, error)
            test_error.append(float(test_error_value))
    print(f"NN train Error {np.mean(train_error)}")
    print(f"NN Test Error {np.mean(test_error)}")

    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def remove_outliers(X,y):

    p95, p5 = np.percentile(y, [95, 5])
    index = [list(y).index(d) for d in y if d<p5 or d>p95]
    X1 = np.delete(X,index,0)
    y1 = np.delete(y,index)
    return X1,y1

def poincare_disk(points):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    # Plot each point in the disk
    for point in points:
        ax.plot(point[0], point[1], 'bo')

    # Draw the boundary of the Poincaré disk
    circle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_artist(circle)

    plt.savefig("hyperbolic.png")


def predict_jepa(train_loader, test_loader,model, rank):
    import pandas as pd
    model.eval()
    X_train, y_train = [], []
    X_test, y_test = [], []
    ### Extracting training features and labels in Scikit-Learn form
    for data in train_loader:
        data.to(rank)
        with torch.no_grad():
            features = model.encode(data)
            X_train.append(features.detach().cpu().numpy())
            y_train.append(data.y.detach().cpu().numpy())

    # Concatenate the lists into numpy arrays
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print("Shapes bef", X_train.shape,y_train.shape)
    # X_train, y_train =remove_outliers(X_train,y_train)
    # print("Shapes aft", X_train.shape, y_train.shape)
    for data in test_loader:
        data.to(rank)
        with torch.no_grad():
            features = model.encode(data)
            X_test.append(features.detach().cpu().numpy())
            y_test.append(data.y.detach().cpu().numpy())

    # Concatenate the lists into numpy arrays
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    print("Shapes bef", X_test.shape, y_test.shape)


    # X_test, y_test = remove_outliers(X_test, y_test)
    # print("Shapes aft", X_test.shape, y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # basic_nn(X_train,y_train,X_test,y_test)
    #
    # print("Data shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # plot_tsne(X_train, y_train, "train")
    plot_tsne(X_test,y_test, "test")
    # exit()
    # # #
    print("Ridge Regression Results")
    # # # Fine tuning on the learned representations via Ridge Regression
    lin_model = Ridge(alpha=0.1)
    # # #lin_model = Lasso()
    lin_model.fit(X_train, y_train)
    lin_score_train = lin_model.score(X_train, y_train)
    lin_score_test = lin_model.score(X_test, y_test)
    print(f"Train Ridge Regression R square.: {lin_score_train}")
    print(f"Test Ridge Regression R square.: {lin_score_test}")
    lin_predictions_train = lin_model.predict(X_train)
    lin_mae_train = mean_absolute_error(y_train, lin_predictions_train)
    lin_predictions_test = lin_model.predict(X_test)
    lin_mae_test = mean_absolute_error(y_test, lin_predictions_test)
    # # print(f'Ridge Regression Coeff.: {lin_model.coef_}')
    # # print(f'Ridge Regression Intercept.: {lin_model.intercept_}')
    print(f'Ridge Regression Train MAE.: {lin_mae_train}')
    print(f'Ridge Regression Test MAE.: {lin_mae_test}')
    #
    #
    knn_model =KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train,y_train)
    knn_predictions_test = knn_model.predict(X_test)
    output_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted KNN': knn_predictions_test,
        'Predicted Linear': lin_predictions_test
    })

    # Save the DataFrame to a CSV file
    output_df.to_csv('model_predictions.csv', index=False)

    knn_mae_test = mean_absolute_error(y_test, knn_predictions_test)
    print(f'KNN Test score.: {knn_mae_test}')


    # fig, ax = plt.subplots()
    # ax.scatter(y_train, lin_predictions_train)
    # lims = [
    #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]
    # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # ax.set_aspect('equal')
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)
    # # ax.axline((0, 0), slope=1)
    # plt.text(3,1.0, f"Train $R^2=${round(lin_model.score(X_train, y_train),2)}")
    # plt.text(3,0.0, f"Test $R^2=${round(lin_model.score(X_test, y_test),2)}")
    # plt.title("Ridge Regression-Formation Energy")
    # plt.xlabel("Actual Formation Energy(eV/atom)")
    # plt.ylabel("Predicted Formation Energy (eV/atom)")
    # plt.savefig("WSe2_fe.png", dpi=300)
    # plt.close()


    #
    # print("Kernel Ridge Regression Results")
    # # Fine tuning on the learned representations via Ridge Regression
    # kernel_model = KernelRidge()
    # kernel_model.fit(X_train, y_train)
    # kernel_predictions = kernel_model.predict(X_test)
    # kernel_mae = mean_absolute_error(y_test, kernel_predictions)
    # print(f'Kernel Ridge Regression MAE.: {kernel_mae}')

    # print("SVM Results")
    # # Fine tuning on the learned representations via Ridge Regression
    # svm_model = SVC(kernel="rbf", C=10)
    # svm_model.fit(X_train, y_train)
    # svm_predictions = svm_model.predict(X_test)
    # svc_accuracy =accuracy_score(y_test, svm_predictions)
    # plt.scatter(y_test,svm_predictions)
    # plt.savefig("Band Gap.png", dpi=300)
    # plt.close()
    # svm_mae = mean_absolute_error(y_test, svm_predictions)
    # print(f'SVM Regression MAE.: {svc_accuracy}')

    import pandas as pd
    #
    # log_model = LogisticRegression(solver='saga', max_iter=1000)
    # log_model.fit(X_train, y_train)
    # log_predictions_train = log_model.predict(X_train)
    # log_predictions_test = log_model.predict(X_test)
    # log_mae_train = accuracy_score(y_train, log_predictions_train)
    # log_mae_test = accuracy_score(y_test, log_predictions_test)
    # # calculate the fpr and tpr for all thresholds of the classification
    # #test_probs = log_model.predict_proba(X_test)
    # test_preds = log_model.predict(X_test)
    # # submission = pd.DataFrame({
    # #     "True values" : list(y_test),
    # #     "PolarPredict": list(test_preds),
    # #     "probabilities": list(test_probs)
    # # })
    #
    # submission.to_csv("logistic_regression_binary_v1.csv", index=False)
    # preds = probs[:, 1]
    # fpr, tpr, threshold = roc_curve(y_test, test_preds)
    # roc_auc = auc(fpr, tpr)
    #
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig("ridge_plot_stability_gcnconv.png", dpi=300)
    # plt.close()
    # print(f'Logistic Regression Coeff.: {log_model.coef_}')
    # print(f'Logistic Regression Intercept.: {log_model.intercept_}')
    # print(f'Logistic Regression Train score.: {log_mae_train}')
    # print(f'Logistic Regression Test score.: {log_mae_test}')

    # knn_model =KNeighborsClassifier(n_neighbors=5)
    # knn_model.fit(X_train,y_train)
    # knn_predictions_test = knn_model.predict(X_test)
    # knn_acc_test = accuracy_score(y_test,  knn_predictions_test)
    # print(f'KNN Test score.: {knn_acc_test}')
    # #

    # from sklearn.neural_network import MLPRegressor
    # ann_model = MLPRegressor(random_state=1, max_iter=500,early_stopping=True, hidden_layer_sizes=(512,256,128, 64),learning_rate_init=0.0001, learning_rate='adaptive', verbose=True)
    # ann_model.fit(X_train,y_train)
    # ann_predictions_train = ann_model.predict(X_train)
    # ann_predictions_test = ann_model.predict(X_test)
    # ann_mae_test = mean_absolute_error(y_test, ann_predictions_test)
    # ann_mae_train= mean_absolute_error(y_train, ann_predictions_train)
    # # ann_mae_test =  accuracy_score(y_test, ann_predictions_test)
    # # ann_mae_train=  accuracy_score(y_train, ann_predictions_train)
    # print(f'ANN Train score.: {ann_mae_train}')
    # print(f'ANN Test score.: {ann_mae_test}')


###Predict using a saved movel
def predict(data_path, training_parameters, model_parameters, job_parameters, loss=None):

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = process.get_dataset(data_path=data_path, target_index=training_parameters["target_index"], reprocess=True,
                                  model_name=model_parameters["model"])

    loss = training_parameters["loss"]
    ##Loads predict dataset in one go, care needed for large datasets)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    ##Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cpu")
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], map_location=torch.device("cuda")
        )
    model = saved["full_model"]
    model = model.to(rank)
    print(model)

    ##Get predictions
    time_start = time.time()
    test_error, test_out = evaluate(loader, model, loss, rank, out=True)
    elapsed_time = time.time() - time_start

    print("Evaluation time (s): {:.5f}".format(elapsed_time))
    print("Test Error: {:.5f}".format(test_error))

    ##Write output
    if job_parameters["write_output"] == "True":
        write_results(
            test_out, str(job_parameters["job_name"]) + "_predicted_outputs.csv"
        )

    return test_error


###n-fold cross validation
def train_CV(
    rank,
    world_size,
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    job_parameters["model_path"] = None
    ##DDP
    ddp_setup(rank, world_size)
    ##some issues with DDP learning rate
    if rank not in ("cpu", "cuda"):
        model_parameters["lr"] = model_parameters["lr"] * world_size


    ##Get dataset
    dataset = process.get_dataset(data_path=data_path, target_index= training_parameters["target_index"], reprocess=False,  model_name = model_parameters["model"])

    ##Split datasets
    cv_dataset = process.split_data_CV(
        dataset, num_folds=job_parameters["cv_folds"], seed=job_parameters["seed"]
    )
    cv_error = 0

    for index in range(0, len(cv_dataset)):

        ##Set up model
        if index == 0:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=True,
            )
        else:
            model = model_setup(
                rank,
                model_parameters["model"],
                model_parameters,
                dataset,
                job_parameters["load_model"],
                job_parameters["model_path"],
                print_model=False,
            )

        ##Set-up optimizer & scheduler
        optimizer = getattr(torch.optim, model_parameters["optimizer"])(
            model.parameters(),
            lr=model_parameters["lr"],
            **model_parameters["optimizer_args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
            optimizer, **model_parameters["scheduler_args"]
        )

        ##Set up loader
        train_loader, test_loader, train_sampler, train_dataset, _ = loader_setup_CV(
            index, model_parameters["batch_size"], cv_dataset, rank, world_size
        )

        ##Start training
        model = trainer(
            rank,
            world_size,
            model,
            optimizer,
            scheduler,
            training_parameters["loss"],
            train_loader,
            None,
            train_sampler,
            model_parameters["epochs"],
            training_parameters["verbosity"],
            "my_model_temp.pth",
        )

        if rank not in ("cpu", "cuda"):
            dist.barrier()

        if rank in (0, "cpu", "cuda"):

            train_loader = DataLoader(
                train_dataset,
                batch_size=model_parameters["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            ##Get train error
            train_error, train_out = evaluate(
                train_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Train Error: {:.5f}".format(train_error))

            ##Get test error
            test_error, test_out = evaluate(
                test_loader, model, training_parameters["loss"], rank, out=True
            )
            print("Test Error: {:.5f}".format(test_error))

            cv_error = cv_error + test_error

            if index == 0:
                total_rows = test_out
            else:
                total_rows = np.vstack((total_rows, test_out))

    ##Write output
    if rank in (0, "cpu", "cuda"):
        if job_parameters["write_output"] == "True":
            if test_loader != None:
                write_results(
                    total_rows, str(job_parameters["job_name"]) + "_CV_outputs.csv"
                )

        cv_error = cv_error / len(cv_dataset)
        print("CV Error: {:.5f}".format(cv_error))

    if rank not in ("cpu", "cuda"):
        dist.destroy_process_group()

    return cv_error


### Repeat training for n times
def train_repeat(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["load_model"] = "False"
    job_parameters["save_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, job_parameters["repeat_trials"]):

        ##new seed each time for different data split
        job_parameters["seed"] = np.random.randint(1, 1e6)

        if i == 0:
            model_parameters["print_model"] = True
        else:
            model_parameters["print_model"] = False

        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = str(i) + "_" + model_path

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters,
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters,
                )

    ##Compile error metrics from individual trials
    print("Individual training finished.")
    print("Compiling metrics from individual trials...")
    error_values = np.zeros((job_parameters["repeat_trials"], 3))
    for i in range(0, job_parameters["repeat_trials"]):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    ##Print error
    print(
        "Training Error Avg: {:.3f}, Training Standard Dev: {:.3f}".format(
            mean_values[0], std_values[0]
        )
    )
    print(
        "Val Error Avg: {:.3f}, Val Standard Dev: {:.3f}".format(
            mean_values[1], std_values[1]
        )
    )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )

    ##Write error metrics
    if job_parameters["write_output"] == "True":
        with open(job_name + "_all_errorvalues.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "",
                    "Training",
                    "Validation",
                    "Test",
                ]
            )
            for i in range(0, len(error_values)):
                csvwriter.writerow(
                    [
                        "Trial " + str(i),
                        error_values[i, 0],
                        error_values[i, 1],
                        error_values[i, 2],
                    ]
                )
            csvwriter.writerow(["Mean", mean_values[0], mean_values[1], mean_values[2]])
            csvwriter.writerow(["Std", std_values[0], std_values[1], std_values[2]])
    elif job_parameters["write_output"] == "False":
        for i in range(0, job_parameters["repeat_trials"]):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)


###Hyperparameter optimization
# trainable function for ray tune (no parallel, max 1 GPU per job)
def tune_trainable(config, checkpoint_dir=None, data_path=None):

    # imports
    from ray import tune

    print("Hyperparameter trial start")
    hyper_args = config["hyper_args"]
    job_parameters = config["job_parameters"]
    processing_parameters = config["processing_parameters"]
    training_parameters = config["training_parameters"]
    model_parameters = config["model_parameters"]

    ##Merge hyperparameter parameters with constant parameters, with precedence over hyperparameter ones
    ##Omit training and job parameters as they should not be part of hyperparameter opt, in theory
    model_parameters = {**model_parameters, **hyper_args}
    processing_parameters = {**processing_parameters, **hyper_args}

    ##Assume 1 gpu or 1 cpu per trial, no functionality for parallel yet
    world_size = 1
    rank = "cpu"
    if torch.cuda.is_available():
        rank = "cuda"

    ##Reprocess data in a separate directory to prevent conflict
    if job_parameters["reprocess"] == "True":
        time = datetime.now()
        processing_parameters["processed_path"] = time.strftime("%H%M%S%f")
        processing_parameters["verbose"] = "False"
    data_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    data_path = os.path.join(data_path, processing_parameters["data_path"])
    data_path = os.path.normpath(data_path)
    print("Data path", data_path)

    ##Set up dataset
    dataset = process.get_dataset(
        data_path,
        training_parameters["target_index"],
        job_parameters["reprocess"],
        processing_parameters,
    )

    ##Set up loader
    (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        train_dataset,
        _,
        _,
    ) = loader_setup(
        training_parameters["train_ratio"],
        training_parameters["val_ratio"],
        training_parameters["test_ratio"],
        model_parameters["batch_size"],
        dataset,
        rank,
        job_parameters["seed"],
        world_size,
    )

    ##Set up model
    model = model_setup(
        rank,
        model_parameters["model"],
        model_parameters,
        dataset,
        False,
        None,
        False,
    )

    ##Set-up optimizer & scheduler
    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(),
        lr=model_parameters["lr"],
        **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    ##Load checkpoint
    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    ##Training loop
    for epoch in range(1, model_parameters["epochs"] + 1):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_error = train(
            model, optimizer, train_loader, training_parameters["loss"], rank=rank
        )

        val_error = evaluate(
            val_loader, model, training_parameters["loss"], rank=rank, out=False
        )

        ##Delete processed data
        if epoch == model_parameters["epochs"]:
            if (
                job_parameters["reprocess"] == "True"
                and job_parameters["hyper_delete_processed"] == "True"
            ):
                shutil.rmtree(
                    os.path.join(data_path, processing_parameters["processed_path"])
                )
            print("Finished Training")

        ##Update to tune
        if epoch % job_parameters["hyper_iter"] == 0:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    path,
                )
            ##Somehow tune does not recognize value without *1
            tune.report(loss=val_error.cpu().numpy() * 1)
            # tune.report(loss=val_error)


# Tune setup
def tune_setup(
    hyper_args,
    job_parameters,
    processing_parameters,
    training_parameters,
    model_parameters,
):

    # imports
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune import CLIReporter

    ray.init()
    data_path = "_"
    local_dir = "ray_results"
    # currently no support for paralleization per trial
    gpus_per_trial = 1

    ##Set up search algo
    search_algo = HyperOptSearch(metric="loss", mode="min", n_initial_points=5)
    search_algo = ConcurrencyLimiter(
        search_algo, max_concurrent=job_parameters["hyper_concurrency"]
    )

    ##Resume run
    if os.path.exists(local_dir + "/" + job_parameters["job_name"]) and os.path.isdir(
        local_dir + "/" + job_parameters["job_name"]
    ):
        if job_parameters["hyper_resume"] == "False":
            resume = False
        elif job_parameters["hyper_resume"] == "True":
            resume = True
        # else:
        #    resume = "PROMPT"
    else:
        resume = False

    ##Print out hyperparameters
    parameter_columns = [
        element for element in hyper_args.keys() if element not in "global"
    ]
    parameter_columns = ["hyper_args"]
    reporter = CLIReporter(
        max_progress_rows=20, max_error_rows=5, parameter_columns=parameter_columns
    )

    ##Run tune
    tune_result = tune.run(
        partial(tune_trainable, data_path=data_path),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config={
            "hyper_args": hyper_args,
            "job_parameters": job_parameters,
            "processing_parameters": processing_parameters,
            "training_parameters": training_parameters,
            "model_parameters": model_parameters,
        },
        num_samples=job_parameters["hyper_trials"],
        # scheduler=scheduler,
        search_alg=search_algo,
        local_dir=local_dir,
        progress_reporter=reporter,
        verbose=job_parameters["hyper_verbosity"],
        resume=resume,
        log_to_file=True,
        name=job_parameters["job_name"],
        max_failures=4,
        raise_on_failed_trial=False,
        # keep_checkpoints_num=job_parameters["hyper_keep_checkpoints_num"],
        # checkpoint_score_attr="min-loss",
        stop={
            "training_iteration": model_parameters["epochs"]
            // job_parameters["hyper_iter"]
        },
    )

    ##Get best trial
    best_trial = tune_result.get_best_trial("loss", "min", "all")
    # best_trial = tune_result.get_best_trial("loss", "min", "last")

    return best_trial


###Simple ensemble using averages
def train_ensemble(
    data_path,
    job_parameters=None,
    training_parameters=None,
    model_parameters=None,
):

    world_size = torch.cuda.device_count()
    job_name = job_parameters["job_name"]
    write_output = job_parameters["write_output"]
    model_path = job_parameters["model_path"]
    job_parameters["write_error"] = "True"
    job_parameters["write_output"] = "True"
    job_parameters["load_model"] = "False"
    ##Loop over number of repeated trials
    for i in range(0, len(job_parameters["ensemble_list"])):
        job_parameters["job_name"] = job_name + str(i)
        job_parameters["model_path"] = (
            str(i) + "_" + job_parameters["ensemble_list"][i] + "_" + model_path
        )

        if world_size == 0:
            print("Running on CPU - this will be slow")
            training.train_regular(
                "cpu",
                world_size,
                data_path,
                job_parameters,
                training_parameters,
                model_parameters[job_parameters["ensemble_list"][i]],
            )
        elif world_size > 0:
            if job_parameters["parallel"] == "True":
                print("Running on", world_size, "GPUs")
                mp.spawn(
                    training.train_regular,
                    args=(
                        world_size,
                        data_path,
                        job_parameters,
                        training_parameters,
                        model_parameters[job_parameters["ensemble_list"][i]],
                    ),
                    nprocs=world_size,
                    join=True,
                )
            if job_parameters["parallel"] == "False":
                print("Running on one GPU")
                training.train_regular(
                    "cuda",
                    world_size,
                    data_path,
                    job_parameters,
                    training_parameters,
                    model_parameters[job_parameters["ensemble_list"][i]],
                )

    ##Compile error metrics from individual models
    print("Individual training finished.")
    print("Compiling metrics from individual models...")
    error_values = np.zeros((len(job_parameters["ensemble_list"]), 3))
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_errorvalues.csv"
        error_values[i] = np.genfromtxt(filename, delimiter=",")
    mean_values = [
        np.mean(error_values[:, 0]),
        np.mean(error_values[:, 1]),
        np.mean(error_values[:, 2]),
    ]
    std_values = [
        np.std(error_values[:, 0]),
        np.std(error_values[:, 1]),
        np.std(error_values[:, 2]),
    ]

    # average ensembling, takes the mean of the predictions
    for i in range(0, len(job_parameters["ensemble_list"])):
        filename = job_name + str(i) + "_test_outputs.csv"
        test_out = np.genfromtxt(filename, delimiter=",", skip_header=1)
        if i == 0:
            test_total = test_out
        elif i > 0:
            test_total = np.column_stack((test_total, test_out[:, 2]))

    ensemble_test = np.mean(np.array(test_total[:, 2:]).astype(np.float), axis=1)
    ensemble_test_error = getattr(F, training_parameters["loss"])(
        torch.tensor(ensemble_test),
        torch.tensor(test_total[:, 1].astype(np.float)),
    )
    test_total = np.column_stack((test_total, ensemble_test))
    
    ##Print performance
    for i in range(0, len(job_parameters["ensemble_list"])):
        print(
            job_parameters["ensemble_list"][i]
            + " Test Error: {:.5f}".format(error_values[i, 2])
        )
    print(
        "Test Error Avg: {:.3f}, Test Standard Dev: {:.3f}".format(
            mean_values[2], std_values[2]
        )
    )
    print("Ensemble Error: {:.5f}".format(ensemble_test_error))
    
    ##Write output
    if write_output == "True" or write_output == "Partial":
        with open(
            str(job_name) + "_test_ensemble_outputs.csv", "w"
        ) as f:
            csvwriter = csv.writer(f)
            for i in range(0, len(test_total) + 1):
                if i == 0:
                    csvwriter.writerow(
                        [
                            "ids",
                            "target",
                        ]
                        + job_parameters["ensemble_list"]
                        + ["ensemble"]
                    )
                elif i > 0:
                    csvwriter.writerow(test_total[i - 1, :])
    if write_output == "False" or write_output == "Partial":
        for i in range(0, len(job_parameters["ensemble_list"])):
            filename = job_name + str(i) + "_errorvalues.csv"
            os.remove(filename)
            filename = job_name + str(i) + "_test_outputs.csv"
            os.remove(filename)

##Obtains features from graph in a trained model and analysis with tsne
def analysis(
    dataset,
    model_path,
    tsne_args,
):

    # imports
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = []

    def hook(module, input, output):
        inputs.append(input)

    assert os.path.exists(model_path), "saved model not found"
    if str(rank) == "cpu":
        saved = torch.load(model_path, map_location=torch.device("cpu"))
    else:
        saved = torch.load(model_path, map_location=torch.device("cuda"))
    model = saved["full_model"]
    print(model)

    print(dataset)

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    ##Grabs the input of the first linear layer after the GNN
    model.post_lin_list[0].register_forward_hook(hook)
    for data in loader:
        with torch.no_grad():
            data = data.to(rank)
            output = model(data)

    inputs = [i for sub in inputs for i in sub]
    inputs = torch.cat(inputs)
    inputs = inputs.cpu().numpy()
    print("Number of samples: ", inputs.shape[0])
    print("Number of features: ", inputs.shape[1])

    # only works for when targets has one index
    targets = dataset.data.y.numpy()

    # pca = PCA(n_components=2)
    # pca_out=pca.fit_transform(inputs)
    # print(pca_out.shape)
    # np.savetxt('pca.csv', pca_out, delimiter=',')
    # plt.scatter(pca_out[:,1],pca_out[:,0],c=targets,s=15)
    # plt.colorbar()
    # plt.show()
    # plt.clf()

    ##Start t-SNE analysis
    tsne = TSNE(**tsne_args)
    tsne_out = tsne.fit_transform(inputs)
    rows = zip(
        dataset.data.structure_id,
        list(dataset.data.y.numpy()),
        list(tsne_out[:, 0]),
        list(tsne_out[:, 1]),
    )

    with open("tsne_output.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        for row in rows:
            writer.writerow(row)

    fig, ax = plt.subplots()
    main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(main, ax=ax)
    stdev = np.std(targets)
    cbar.mappable.set_clim(
        np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
    )
    # cbar.ax.tick_params(labelsize=50)
    # cbar.ax.tick_params(size=40)
    plt.savefig("tsne_output.png", format="png", dpi=600)
    plt.show()


def plot_tsne(input, targets, name):
    tsne = TSNE()
    tsne_out = tsne.fit_transform(input)
    fig, ax = plt.subplots()
    main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(main, ax=ax)
    # stdev = np.std(targets)
    cbar.mappable.set_clim(
        np.mean(targets) - 2 * np.std(targets), np.mean(targets) + 2 * np.std(targets)
    )
    # cbar.ax.tick_params(labelsize=50)
    # cbar.ax.tick_params(size=40)
    plt.savefig(f"bg_tsne_output_{name}.png", format="png", dpi=300)
    plt.close()