import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt  # plotting package
import glob
from tqdm import tqdm


storage_path = "/mnt/storage/lborgna/track/"
input_files = glob.glob(storage_path + "??.root")

# input_files = input_files[0:2]

n_files = len(input_files)

fig_trk_pt, ax_trk_pt = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))

fig_trk_z0, ax_trk_z0 = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))

fig_tp_z0, ax_tp_z0 = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))

fig_true_z0, ax_true_z0 = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))

fig_res, ax_res = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))

fig_res_match, ax_res_match = plt.subplots(n_files, 1, figsize=(8, 6 * n_files))


for i, file in enumerate(tqdm(input_files)):
    # print(file)
    file_num = file[-7:-5]
    f = uproot.open(file)
    events = f[b"L1TrackNtuple/eventTree"]
    n_events = events.numentries
    # Define useful variables, masks, and true z0 of primary vertex
    tp_vertex_id = events["tp_eventid"].array()
    tp_z0 = events["tp_z0"].array()
    tp_d0 = events["tp_d0"].array()
    trk_z0 = events["trk_z0"].array()
    trk_pt = events["trk_pt"].array()
    mask = tp_vertex_id == 0
    mask_no_sv = np.abs(tp_d0) < 0.01
    true_z0 = tp_z0[mask & mask_no_sv].mean()
    matchtrk_z0 = events["matchtrk_z0"].array()

    # Plot the trk_pt distribution
    counts, edges = np.histogram(trk_pt.flatten(), bins=75, range=(0.0, 150.0))
    ax_trk_pt[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_trk_pt[i].set_xlim(edges[0], edges[-1])
    ax_trk_pt[i].set_yscale("log")
    ax_trk_pt[i].set_ylim(0.5, counts.max() * 2.0)
    ax_trk_pt[i].set_xlabel("Track transverse momentum (GeV)")
    ax_trk_pt[i].set_ylabel("Counts/bin")
    ax_trk_pt[i].set_title(f"{file_num}.root, {n_events} events")

    # Plot the trk_z0 distribution
    counts, edges = np.histogram(trk_z0.flatten(), bins=20, range=(-5.0, 5.0))
    ax_trk_z0[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_trk_z0[i].set_xlim(edges[0], edges[-1])
    ax_trk_z0[i].set_yscale("linear")
    ax_trk_z0[i].set_ylim(0.0, counts.max() * 2.0)
    ax_trk_z0[i].set_xlabel("Track z0 position (cm)")
    ax_trk_z0[i].set_ylabel("Counts/bin")
    ax_trk_z0[i].set_title(f"{file_num}.root, {n_events} events")

    # Plot the tp_z0 distribution
    counts, edges = np.histogram(tp_z0.flatten(), bins=20, range=(-5.0, 5.0))
    ax_tp_z0[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_tp_z0[i].set_xlim(edges[0], edges[-1])
    ax_tp_z0[i].set_yscale("linear")
    ax_tp_z0[i].set_ylim(0.0, counts.max() * 2.0)
    ax_tp_z0[i].set_xlabel("TP z0 position (cm)")
    ax_tp_z0[i].set_ylabel("Counts/bin")
    ax_tp_z0[i].set_title(f"{file_num}.root, {n_events} events")

    # Plot the true z0 position of the primary vertex
    counts, edges = np.histogram(true_z0.flatten(), bins=10, range=(-5.0, 5.0))
    ax_true_z0[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_true_z0[i].set_xlim(edges[0], edges[-1])
    ax_true_z0[i].set_yscale("linear")
    ax_true_z0[i].set_ylim(0.0, counts.max() * 2.0)
    ax_true_z0[i].set_xlabel("True primary vertex z0 position (cm)")
    ax_true_z0[i].set_ylabel("Counts/bin")
    ax_true_z0[i].set_title(f"{file_num}.root, {n_events} events")

    # Plot the residuals between the trk_z0 and true_z0
    residuals = trk_z0 - true_z0
    counts, edges = np.histogram(residuals.flatten(), bins=100, range=(-5.0, 5.0))
    ax_res[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_res[i].set_xlim(edges[0], edges[-1])
    ax_res[i].set_yscale("linear")
    ax_res[i].set_ylim(0.0, counts.max() * 1.2)
    ax_res[i].set_xlabel("Track z0 position - (true) primary vertex z0 position (cm)")
    ax_res[i].set_ylabel("Counts/bin")
    ax_res[i].set_title(f"{file_num}.root, {n_events} events")

    # The above plot shows a strong peak at zero, which is due to the
    # tracks originating from the primary vertex. The peak is superimposed
    # on an almost-flat continuous background, which is due to tracks from
    # the other interactions that happen in an LHC event (called pileup).

    # Plot the residuals between the matchtrk_z0 and tp_z0
    residuals = matchtrk_z0 - tp_z0
    counts, edges = np.histogram(residuals.flatten(), bins=100, range=(-5.0, 5.0))
    ax_res_match[i].step(x=edges, y=np.append(counts, 0), where="post")
    ax_res_match[i].set_xlim(edges[0], edges[-1])
    ax_res_match[i].set_yscale("linear")
    ax_res_match[i].set_ylim(0.0, counts.max() * 1.2)
    ax_res_match[i].set_xlabel(
        "Matched track z0 position - (true) primary vertex z0 position (cm)"
    )
    ax_res_match[i].set_ylabel("Counts/bin")
    ax_res_match[i].set_title(f"{file_num}.root, {n_events} events")

fig_trk_pt.savefig("plots/trk_pt.pdf")
fig_trk_z0.savefig("plots/trk_z0.pdf")
fig_tp_z0.savefig("plots/tp_z0.pdf")
fig_true_z0.savefig("plots/true_z0.pdf")
fig_res.savefig("plots/residuals_all.pdf")
fig_res_match.savefig("plots/residuals.pdf")

# The above plot shows a strong peak at zero, which is due to the
# tracks matched to TPs that originate from the primary vertex.
