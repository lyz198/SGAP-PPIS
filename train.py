import os
import sys
import time
import pickle
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import KFold
from ppis_core.sgha_ppis_model import *


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(default_name, env_name):
    env_value = os.environ.get(env_name, "")
    if env_value:
        if os.path.isabs(env_value):
            return os.path.normpath(env_value)
        return os.path.normpath(os.path.join(PROJECT_DIR, env_value))
    return os.path.normpath(os.path.join(PROJECT_DIR, default_name))


# Path
Dataset_Path = resolve_project_path("Dataset", "HHGA_DATASET_DIR")
Model_Path = resolve_project_path("Model", "HHGA_MODEL_DIR")
Log_path = resolve_project_path("Log", "HHGA_LOG_DIR")
model_time = None
VALID_THRESHOLD = 0.5  # Set to None to use oracle-threshold selection on validation.


def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0

    for data in data_loader:
        model.optimizer.zero_grad()
        _, _, labels, node_features, G_batch, adj_matrix, pos = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda().float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            G_batch = G_batch.to(torch.device('cuda:0'))
            adj_matrix = Variable(adj_matrix.cuda())
            y_true = Variable(labels.cuda())
            pos = Variable(pos.cuda().float())
        else:
            node_features = Variable(node_features.float())
            G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
            adj_matrix = Variable(adj_matrix)
            y_true = Variable(labels)
            pos = Variable(pos.float())

        adj_matrix = torch.squeeze(adj_matrix)
        y_true = torch.squeeze(y_true).long()

        y_pred = model(node_features, G_batch, adj_matrix, pos)
        loss = model.criterion(y_pred, y_true)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss detected during training.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n += 1

    return epoch_loss_train / n


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix, pos = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())
                pos = Variable(pos.cuda().float())
            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                pos = Variable(pos.float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true).long()

            y_pred = model(node_features, G_batch, adj_matrix, pos)
            loss = model.criterion(y_pred, y_true)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss detected during evaluation.")

            y_prob = torch.softmax(y_pred, dim=1).cpu().detach().numpy()
            if not np.isfinite(y_prob).all():
                raise FloatingPointError("Non-finite probabilities detected during evaluation.")
            y_true = y_true.cpu().detach().numpy()

            valid_pred += [pred[1] for pred in y_prob]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_prob]

            epoch_loss += loss.item()
            n += 1

    return epoch_loss / n, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold=None):
    if not np.isfinite(np.asarray(y_pred, dtype=np.float64)).all():
        raise FloatingPointError("Non-finite prediction scores passed to metrics analysis.")
    if best_threshold is None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            f1 = metrics.f1_score(y_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]

    binary_acc = metrics.accuracy_score(y_true, binary_pred)
    precision = metrics.precision_score(y_true, binary_pred)
    recall = metrics.recall_score(y_true, binary_pred)
    f1 = metrics.f1_score(y_true, binary_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_pred)
    auprc = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(y_true, binary_pred)

    return {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': auc,
        'AUPRC': auprc,
        'mcc': mcc,
        'threshold': best_threshold,
    }


def train(model, train_dataframe, valid_dataframe, fold=0):
     #  4. 记录全量训练开始时间
    start_time = time.time()
    train_loader = DataLoader(
        dataset=ProDataset(train_dataframe),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=graph_collate,
    )
    valid_loader = DataLoader(
        dataset=ProDataset(valid_dataframe),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=graph_collate,
    )

    best_epoch = 0
    best_result = None
    best_val_aupr = -1.0

    for epoch in range(NUMBER_EPOCHS):
       
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("Train loss: ", epoch_loss_train_avg)

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred, VALID_THRESHOLD)

        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid binary acc: ", result_valid['binary_acc'])
        print("Valid precision: ", result_valid['precision'])
        print("Valid recall: ", result_valid['recall'])
        print("Valid f1: ", result_valid['f1'])
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        print("Valid mcc: ", result_valid['mcc'])
        print("Valid threshold: ", result_valid['threshold'])

        if result_valid['AUPRC'] > best_val_aupr:
            best_epoch = epoch + 1
            best_result = result_valid
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

        model.scheduler.step(result_valid['AUPRC'])
    #  2. 计算并打印当前 Fold 的总训练时长
    end_time = time.time()
    print(f"\n========== Fold {fold} Training Time: {end_time - start_time:.2f} seconds ==========")

    if best_result is None:
        return 0, 0.5, 0.0
    return best_epoch, best_result['AUC'], best_result['AUPRC']


def cross_validation(all_dataframe, fold_number=5):
    print("Architecture:", ARCH_NAME)
    print("Random seed:", SEED)
    print("Add node features:", ADD_NODEFEATS)
    print("Map cutoff:", MAP_CUTOFF)
    print("The parameter of normalizing the distance:", DIST_NORM)
    print("Feature dim:", INPUT_DIM)
    print("Hidden dim1:", HIDDEN_DIM1)
    print("Hidden dim2:", HIDDEN_DIM2)
    print("Total layers (stack blocks): N/A (MGMA-style single block)")
    print("APPNP branch layers (replacing EGAT):", APPNP_LAYER)
    print("EGNN layers:", EGNN_LAYER)
    print("Hybrid layers:", "N/A")
    print("APPNP propagation steps:", APPNP_LAYER)
    print("Dropout:", DROPOUT)
    print("Learning rate:", LEARNING_RATE)
    print("Training epochs:", NUMBER_EPOCHS)
    print("Validation threshold:", "oracle" if VALID_THRESHOLD is None else VALID_THRESHOLD)
    print()

    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values

    kfold = KFold(n_splits=fold_number, shuffle=True)
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    fold = 0
    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print(
            "Train on",
            str(train_dataframe.shape[0]),
            "samples, validate on",
            str(valid_dataframe.shape[0]),
            "samples",
        )

        model = SGHAPPIS(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
        #  3. 统计并打印模型的参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"========== Model Parameters: {total_params:,} ==========")
        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, aver_epoch):
    #  4. 记录全量训练开始时间
    start_time = time.time()
    print("\n\nTraining a full model using all training data...\n")
    model = SGHAPPIS(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
    # 🌟 5. 统计并打印全量训练时的模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"========== Full Model Parameters: {total_params:,} ==========")
    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(
        dataset=ProDataset(all_dataframe),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=graph_collate,
    )

    best_tra_aupr = -1.0
    snapshot_epochs = [ep for ep in sorted(set([int(aver_epoch), 40, 45])) if ep <= NUMBER_EPOCHS]

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("Train loss: ", epoch_loss_train_avg)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred, _ = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, VALID_THRESHOLD)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        if result_train['AUPRC'] > best_tra_aupr:
            best_tra_aupr = result_train['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model.pkl'))

        if (epoch + 1) in snapshot_epochs:
            torch.save(model.state_dict(), os.path.join(Model_Path, f'Full_model_{epoch + 1}.pkl'))
        # 🌟 6. 计算并打印全量模型的总训练时长
    end_time = time.time()
    print(f"\n========== Full Model Training Time: {end_time - start_time:.2f} seconds ==========")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def main():
    if not os.path.exists(Log_path):
        os.makedirs(Log_path)

    with open(os.path.join(Dataset_Path, "Train_335.pkl"), "rb") as f:
        Train_335 = pickle.load(f)
        if '2j3rA' in Train_335:
            Train_335.pop('2j3rA')

    IDs, sequences, labels = [], [], []
    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dataframe = pd.DataFrame({"ID": IDs, "sequence": sequences, "label": labels})
    aver_epoch = cross_validation(train_dataframe, fold_number=5)
    train_full_model(train_dataframe, aver_epoch)


if __name__ == "__main__":
    if model_time is not None:
        checkpoint_path = os.path.normpath(os.path.join(Log_path, model_time))
    else:
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        checkpoint_path = os.path.normpath(os.path.join(Log_path, localtime))
        os.makedirs(checkpoint_path)

    Model_Path = os.path.normpath(os.path.join(checkpoint_path, "model"))
    if not os.path.exists(Model_Path):
        os.makedirs(Model_Path)

    sys.stdout = Logger(os.path.normpath(os.path.join(checkpoint_path, "training.log")))
    main()
    sys.stdout.log.close()
