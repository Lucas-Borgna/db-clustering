import uproot
import pandas as pd
import glob


def convert():
    storage_path = "/mnt/storage/lborgna/track/"
    input_files = glob.glob(storage_path + "??.root")

    columns = ['tp_pt','tp_z0','tp_d0','tp_eventid','trk_z0','trk_pt']

    list_tp_df = []
    list_trk_df = []
    for input_file in input_files:
        print(input_file)

        file_number = input_file[-7:-5]
        output_file_tp = storage_path + f"tp_{file_number}.pkl"
        output_file_trk = storage_path + f"trk_{file_number}.pkl"

        _f = uproot.open(input_file)
        _t = _f["L1TrackNtuple/eventTree;1"]

        dfs = _t.arrays(columns, library='pd')

        dfs[0].to_pickle(output_file_tp)
        dfs[1].to_pickle(output_file_trk)
        
        # list_tp_df.append(dfs[0])
        # list_trk_df.append(dfs[1])

    # tp = pd.concat(list_tp_df)
    # trk = pd.concat(list_trk_df)

if __name__ == "__main__":

    convert()
