import os
import pickle
import pandas as pd
from pathlib import Path

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn import metrics

from ppis_core.sgha_ppis_model import *


PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)


def resolve_project_path(default_name, env_name):
    env_value = os.environ.get(env_name, "")
    if env_value:
        env_path = Path(env_value)
        if env_path.is_absolute():
            return env_path
        return PROJECT_DIR / env_path
    return PROJECT_DIR / default_name


Dataset_Path = resolve_project_path("Dataset", "HHGA_DATASET_DIR")


def resolve_model_path():
    
    env_path = os.environ.get("HHGA_MODEL_PATH", "")
    if env_path and os.path.isdir(env_path):
        return env_path

    default_model_dir = resolve_project_path("Model", "HHGA_MODEL_DIR")
    if default_model_dir.is_dir() and len(os.listdir(default_model_dir)) > 0:
        return str(default_model_dir)

    default_log_dir = resolve_project_path("Log", "HHGA_LOG_DIR")
    if not default_log_dir.is_dir():
        raise FileNotFoundError(f"Cannot find {default_log_dir} or {default_model_dir} for checkpoints.")

    subdirs = [d for d in os.listdir(default_log_dir) if os.path.isdir(os.path.join(default_log_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoints found under {default_log_dir}.")

    latest = sorted(subdirs)[-1]
    model_path = os.path.join(default_log_dir, latest, "model")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
   
    return model_path
 
    #return  "/home/zhuenqiang_2/HHGA-PPIS/HHGA-PPIS/Log/2026-03-07-01-48-42/model"
  

Model_Path = None





def _infer_checkpoint_arch(state_dict):
    
    h1 = int(state_dict["deep_agat.fcs.0.weight"].shape[0])
    h2 = int(state_dict["deep_agat.fcs.1.weight"].shape[0])

   
    if "deep_agat.fcs.2.weight" in state_dict:
        fusion_dim = int(state_dict["deep_agat.fcs.2.weight"].shape[1])
    else:
        fusion_dim = int(state_dict["deep_agat.sa.fc_q.weight"].shape[0])
    appnp_layers = max(1, fusion_dim // h1 - 1)

  
    if "deep_agat.egnn_scale_pool_logits" in state_dict:
        egnn_layers = int(state_dict["deep_agat.egnn_scale_pool_logits"].shape[0])
    else:
        proj_ids = []
        for k in state_dict.keys():
            if k.startswith("deep_agat.egnn_scale_projs."):
                parts = k.split(".")
                if len(parts) >= 4 and parts[2].isdigit():
                    proj_ids.append(int(parts[2]))
        egnn_layers = (max(proj_ids) + 1) if proj_ids else EGNN_LAYER

    return {
        "hidden_dim1": h1,
        "hidden_dim2": h2,
        "appnp_layers": appnp_layers,
        "egnn_layers": egnn_layers,
    }


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

            y_prob = torch.softmax(y_pred, dim=1).cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()

            valid_pred += [pred[1] for pred in y_prob]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_prob]

            epoch_loss += loss.item()
            n += 1

    return epoch_loss / n, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold=None):
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
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_pred)

    return {
        'binary_acc': metrics.accuracy_score(y_true, binary_pred),
        'precision': metrics.precision_score(y_true, binary_pred),
        'recall': metrics.recall_score(y_true, binary_pred),
        'f1': metrics.f1_score(y_true, binary_pred),
        'AUC': metrics.roc_auc_score(y_true, y_pred),
        'AUPRC': metrics.auc(recalls, precisions),
        'mcc': metrics.matthews_corrcoef(y_true, binary_pred),
        'threshold': best_threshold,
    }


def test(test_dataframe, psepos_path):
    global Model_Path
    if Model_Path is None:
        Model_Path = resolve_model_path()


    test_loader = DataLoader(
        dataset=ProDataset(dataframe=test_dataframe, psepos_path=psepos_path),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=graph_collate,
    )

    model_files = sorted([f for f in os.listdir(Model_Path) if f.endswith(".pkl")])
    if not model_files:
        raise FileNotFoundError(f"No *.pkl checkpoints found in {Model_Path}.")

    for model_name in model_files:
        print(model_name)
        ckpt_path = os.path.join(Model_Path, model_name)
        state_dict = torch.load(ckpt_path, map_location=device)
        arch = _infer_checkpoint_arch(state_dict)
        print(
            "Checkpoint arch:",
            f"H1={arch['hidden_dim1']}, H2={arch['hidden_dim2']}, "
            f"APPNP={arch['appnp_layers']}, EGNN={arch['egnn_layers']}",
        )
        model = SGHAPPIS(
            INPUT_DIM,
            arch["hidden_dim1"],
            arch["hidden_dim2"],
            NUM_CLASSES,
            DROPOUT,
            LAMBDA,
            ALPHA,
            egat_layers=arch["appnp_layers"],
            egnn_layers=arch["egnn_layers"],
        )
        if torch.cuda.is_available():
            model.cuda()
        load_ret = model.load_state_dict(state_dict, strict=False)
        if load_ret.missing_keys or load_ret.unexpected_keys:
            print("Load warning: missing keys =", len(load_ret.missing_keys),
                  "unexpected keys =", len(load_ret.unexpected_keys))

        epoch_loss_test_avg, test_true, test_pred, _ = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])


def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])

    test_dataframe = pd.DataFrame({"ID": IDs, "sequence": sequences, "label": labels})
    test(test_dataframe, psepos_path)


def main():
    with open(Dataset_Path / "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path / "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path / "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    Test60_psepos_Path = str(PROJECT_DIR / "Feature" / "psepos" / "Test60_psepos_SC.pkl")
    Test315_28_psepos_Path = str(PROJECT_DIR / "Feature" / "psepos" / "Test315-28_psepos_SC.pkl")
    UBtest31_28_psepos_Path = str(PROJECT_DIR / "Feature" / "psepos" / "UBtest31-6_psepos_SC.pkl")

    print("HHGA_MODEL_PATH =",  resolve_model_path())
    print(f"Evaluate {MODEL_NAME} on Test_60")
    test_one_dataset(Test_60, Test60_psepos_Path)

    # Optional external tests.
    print(f"Evaluate {MODEL_NAME} on Test_315-28")
    test_one_dataset(Test_315_28, Test315_28_psepos_Path)

    print(f"Evaluate {MODEL_NAME} on UBtest_31-6")
    test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path)


if __name__ == "__main__":
    main()
