if __name__ == "__main__":
    basecmd ="python src/004_run_experiment.py -F {} -L {}"
    for labelfile in ["labels/last6m-median.csv",
                      "labels/label2outof3slopes.csv"]:
        feature_list = []
        for featurefile in ["features/first6m-rev.csv",
                            "features/distid.csv",
                            #"features/network.csv",
                            "features/unweighted-nw.csv",
                            "features/poi.csv",
                            "features/demographic.csv",
                            "features/mccid.csv",
                            "features/f6month-slopes.csv"]:
            feature_list.append(featurefile)
            cmd = basecmd.format(" ".join(feature_list), labelfile)
            print(cmd)

        for featurefile in ["features/unweighted-nw.csv",
                            "features/weighted-nw.csv",
                            "features/network.csv"]:
            cmd = basecmd.format(featurefile, labelfile)
            print(cmd)

