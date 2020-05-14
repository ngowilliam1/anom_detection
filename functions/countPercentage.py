import numpy as np
import pandas as pd
import pickle 
datasets = ['credit','kdd','mammography','seismic']

for dataset_name in datasets:
    print(f"Currently DS: {dataset_name}")
    if dataset_name == 'credit':
        name_of_label_var = "Class"
    elif dataset_name == 'kdd':
        name_of_label_var = "label"
    elif dataset_name == 'mammography' or dataset_name == 'seismic':
        name_of_label_var = "class"
    
    if dataset_name == 'kdd':
        pathOfDS = '../data/kddcup.data.corrected'
        col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label"]
        df = pd.read_csv(pathOfDS, header=None, names=col_names, index_col=False)
    elif dataset_name == 'credit' or dataset_name == 'mammography' or dataset_name == 'seismic' :
        pathOfDS = f'../data/{dataset_name}.csv'
        df = pd.read_csv(pathOfDS, low_memory=False, index_col=False).rename(columns={name_of_label_var: "label"})

    # Changing normal label from -1 to 0
    if dataset_name == 'mammography':
        df['label'] = df['label'].replace(-1,0)

    if dataset_name == 'credit' or dataset_name == 'seismic' or dataset_name == 'mammography':
        normal = 0
    elif dataset_name == 'kdd':
        normal = 'normal.'
    labels = df['label'].copy()
    is_anomaly = labels != normal
    an_mean = is_anomaly.sum() / is_anomaly.count()
    an_sum = is_anomaly.sum()
    print("Initial Anomalies is: ",an_sum)
    print("Initial Normals is: ", (labels == normal).sum())
    print("Initial Count is: ", is_anomaly.count())
    print("Initial Percent Anomaly is: ", an_mean)