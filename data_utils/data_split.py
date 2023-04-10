# Python file to split training data annotations to train and validation
# algorithm as used in: https://github.com/arkel23/Pretrained-ViT/blob/main/tools/preprocess/data_split.py







import pandas as pd
import os
import argparse






def data_split(args):
    """
    Function to split csv annotations of original training to training and validation annotations.
    

    Args:
        csv_path (str): path to the original annotation csv file.
        train_ratio (float): percentage of original data to be used for training. Default: 0.8
        save_train (str): name of the new train annotation file. Default: None
        save_val (str): name of the new validation annotation file. Default: None
    """
    df = pd.read_csv(args.csv_path)
    print('Length of original df: ', len(df))

    n_per_class_df = df.groupby('class_id', as_index = True).count()

    df_list_train, df_list_val = [], []

    for class_id, n_per_class in enumerate(n_per_class_df['name']):
        train_samples_class = int(n_per_class * args.train_ratio)
        val_samples_class = n_per_class - train_samples_class

        assert(train_samples_class + val_samples_class == n_per_class)

        train_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').head(train_samples_class)
        val_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').tail(val_samples_class)
        df_list_train.append(train_subset_class)
        df_list_val.append(val_subset_class)

    df_train = pd.concat(df_list_train)
    df_val = pd.concat(df_list_val)

    print('Train df: ')
    print(df_train.head())
    print(f"Images in new training set: {df_train.shape}")
    print('Validation df: ')
    print(df_val.head())
    print(f"Images in new validation set: {df_val.shape}")

    root_path = os.path.split(os.path.normpath(args.csv_path))[0]

    train_name = args.save_train if args.save_train is not None else "train.csv"
    train_save_path = os.path.join(root_path, train_name)
    df_train.to_csv(train_save_path, sep = ",", header = True, index = False)
    print(f"Saved {train_name} in {train_save_path}.")

    val_name = args.save_val if args.save_val is not None else "val.csv"
    val_save_path = os.path.join(root_path, val_name)
    df_val.to_csv(val_save_path, sep = ",", header = True, index = False)
    print(f"Saved {val_name} in {val_save_path}.")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type = str, help = "path to the original annotation csv file", required = True)
    parser.add_argument("--train_ratio", type = float, help = "percentage of original data to be used for training", default = 0.8)
    parser.add_argument('--save_train', type = str, help = "name of the new train annotation file", default = None)
    parser.add_argument('--save_val', type = str, help = "name of the new validation annotation file", default = None)

    args = parser.parse_args()

    data_split(args)