import pandas as pd
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    df = pd.read_csv("output/000_filtered_merchant_trans_df.csv")
    df = df.drop_duplicates(
        "UYEISYERI_ID_MASK").set_index(
        "UYEISYERI_ID_MASK")["ISYERI_TURU"]
    onehot_enc = OneHotEncoder()
    X_mccid = onehot_enc.fit_transform(
        df.as_matrix().reshape(-1, 1)).todense()
    X_mccid_df = pd.DataFrame(X_mccid,
                               columns=onehot_enc.active_features_)
    X_mccid_df.index = df.index
    X_mccid_df.to_csv("features/mccid.csv")

