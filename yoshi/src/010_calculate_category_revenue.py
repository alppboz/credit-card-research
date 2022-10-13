import pandas as pd

if __name__ == "__main__":
    mcc_df = pd.read_csv("data/BKM_MCC_ACIKLAMA.csv")
    trans_df = pd.read_csv("output/000_filtered_merchant_trans_df.csv")
    trans_df = pd.merge(trans_df, mcc_df[["description", "mcc"]],
                        left_on="ISYERI_TURU",
                        right_on="mcc")
    agg_df = trans_df.groupby("UYEISYERI_ID_MASK").agg({"ISLEM_TUTARI": "sum"})

    merchantid_mcc_df = trans_df.drop_duplicates(
        subset=["UYEISYERI_ID_MASK"]).set_index(
        "UYEISYERI_ID_MASK")[["mcc", "description"]]

    merge_df = pd.merge(agg_df,
                        merchantid_mcc_df,
                        left_index=True,
                        right_index=True,
                        how="left")

    bigcat_median_revenue_df = merge_df.groupby(
        ["description"]).agg(
        {"ISLEM_TUTARI": ["median", "std", "count"]}).sort_values(
        ("ISLEM_TUTARI", "median"), ascending=False)

    bigcat_median_revenue_df.to_csv(
        "output/010_bigcategory_revenue_stats.csv")