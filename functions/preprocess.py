import time
import numpy as np
import pandas as pd
import pickle
from functions import utilities
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def data_proc(dataset_name: str, perc: float, scaling: str = "min_max", cat_vars: list=None, cat_incl: bool= True) -> dict:
    start = time.time()
    
    if dataset_name == 'credit':
        name_of_label_var = "Class"
    elif dataset_name == 'kdd':
        name_of_label_var = "label"
    elif dataset_name == 'mammography':
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
    elif dataset_name == 'credit' or dataset_name == 'mammography' :
        pathOfDS = f'../data/{dataset_name}.csv'
        df = pd.read_csv(pathOfDS, low_memory=False, index_col=False).rename(columns={name_of_label_var: "label"})

    # Changing normal label from -1 to 0
    if dataset_name == 'mammography':
        df['label'] = df['label'].replace(-1,0)

    le = LabelEncoder()
    le.fit(df.label)
    df_new = reduce_anomalies(df, dataset_name=dataset_name, pct_anomalies=perc)
    
    # find unique labels for each categorical var
    if dataset_name == 'kdd' or dataset_name == 'seismic':
        cat_data = pd.get_dummies(df_new[cat_vars])
        numeric_vars = list(set(df_new.columns.values.tolist()) - set(cat_vars))
        numeric_data = df_new[numeric_vars].copy()
    elif dataset_name == 'credit' or dataset_name == 'mammography':
        cat_data = None
        numeric_vars = df_new.columns.values.tolist()
        
    numeric_data = df_new[numeric_vars].copy().drop('label',axis=1)
    
    if cat_incl:
    # concat numeric and the encoded categorical variables
        numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)
    else:
        numeric_cat_data = numeric_data

    # here we do a quick sanity check that the data has been concatenated correctly by checking the dimension of the vectors
    print(f'cat_data shape:{cat_data.shape if cat_data is not None else None}')
    print(f'numeric_data:{numeric_data.shape}')
    print(f'numeric_cat_data:{numeric_cat_data.shape}')
    
    # capture the labels
    labels = df_new['label'].copy()

    if dataset_name == 'kdd':
        labels = le.transform(labels)
        binary_labels = utilities.convert_label_to_binary(dataset_name,le,labels)
    elif dataset_name == 'credit' or dataset_name == 'mammography' or dataset_name == 'seismic':
        binary_labels = df_new['label'].copy()


    # split data into train, valid, test (~ 70/15/15)
    x_train, x_test, y_train, y_test = train_test_split(numeric_cat_data,
                                                        binary_labels,
                                                        test_size=.15,
                                                        random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          random_state=1)
                                                          
    if scaling == "standard":
        # Scale the data using the StandardScaler from the scikit learn package
        stardard_scaler = StandardScaler()
        stardard_scaler.fit(x_train)
        x_train_standard = stardard_scaler.transform(x_train).astype(np.float32)
        x_valid_standard = stardard_scaler.transform(x_valid).astype(np.float32)
        x_test_standard = stardard_scaler.transform(x_test).astype(np.float32)

        preprocessed_data = {
            'x_train': x_train_standard,
            'y_train': y_train,
            'x_valid': x_valid_standard,
            'y_valid': y_valid,
            'x_test': x_test_standard,
            'y_test': y_test,
            'le': le
        }
        return {"pp_standard":preprocessed_data, "runtime":time.time()-start}

    elif scaling == "min_max":
        # Scale the data using the MinMaxScaler from the scikit learn package
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(x_train)
        x_train_min_max = min_max_scaler.transform(x_train).astype(np.float32)
        x_valid_min_max = min_max_scaler.transform(x_valid).astype(np.float32)
        x_test_min_max = min_max_scaler.transform(x_test).astype(np.float32)

        preprocessed_data = {
            'x_train': x_train_min_max,
            'y_train': y_train,
            'x_valid': x_valid_min_max,
            'y_valid': y_valid,
            'x_test': x_test_min_max,
            'y_test': y_test,
            'le': le
        }
        return {"pp_min_max":preprocessed_data, "runtime":time.time()-start}



def reduce_anomalies(df: pd.core.frame.DataFrame, dataset_name: str, pct_anomalies: float) -> pd.core.frame.DataFrame:
    #Method used to undersample data to achieve a desired anomaly rate

    labels = df['label'].copy()
    
    #Determining what a "normal case is"
    if dataset_name == 'credit' or dataset_name == 'seismic' or dataset_name == 'mammography':
        normal = 0
    elif dataset_name == 'kdd':
        normal = 'normal.'

    is_anomaly = labels != normal
    an_mean = is_anomaly.sum() / is_anomaly.count()
    an_sum = is_anomaly.sum()
    print("Initial Anomalies is: ",an_sum)
    print("Initial Normals is: ", (labels == normal).sum())
    print("Initial Count is: ", is_anomaly.count())
    print("Initial Percent Anomaly is: ", an_mean)
    print("Target Anomaly Rate is: ", pct_anomalies)
    if pct_anomalies >= an_mean:
        print("Need to undersample our non-anomalies")
        num_normals = int(an_sum * ((1.0 / pct_anomalies) - 1))
        all_normals = labels[labels == normal]
        normals_to_keep = np.random.choice(all_normals.index, size=num_normals, replace=False)
        normal_data = df.iloc[normals_to_keep].copy()
        anomalous_data = df[is_anomaly].copy()
    if pct_anomalies < an_mean:
        print("Need to undersample our anomalies")
        num_normal = np.sum(~is_anomaly)
        num_anomalies = int(num_normal/((1.0/pct_anomalies)-1.0))
        all_anomalies = labels[labels != normal]
        anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
        anomalous_data = df.iloc[anomalies_to_keep].copy()
        normal_data = df[~is_anomaly].copy()

    new_df = pd.concat([normal_data, anomalous_data], axis=0)
    labels = new_df['label'].copy()
    is_anomaly = labels != normal
    an_mean = is_anomaly.sum() / is_anomaly.count()
    an_sum = is_anomaly.sum()
    print("Resultant Anomalies is: ", an_sum)
    print("Initial Normals is: ", (labels == normal).sum())
    print("Resultant Count is: ", is_anomaly.count())
    print("Resultant Percent Anomaly is: ", an_mean)
    print("Target Anomaly Rate is: ", pct_anomalies)

    return new_df