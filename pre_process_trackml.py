import numpy as np
import pandas as pd
import sys


import argparse

import glob
import logging

logging.basicConfig(level=logging.DEBUG)


def calc_eta(r: np.array, z: np.array) -> np.array:
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def calc_r(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance of each hit from the origin.
    """
    r = np.sqrt(hits.x**2 + hits.y**2) / 10
    return r


def calc_pt(particles: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the transverse momentum of each particle.
    """
    pt = np.sqrt(particles.px**2 + particles.py**2)
    return pt


def calc_z0(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the z-intercept of the track in r-z plane.
    """
    list_of_particles = hits["particle_id"].unique()
    z0s = []

    for particle in list_of_particles:
        r = hits.loc[hits["particle_id"] == particle, "r"].values
        z = hits.loc[hits["particle_id"] == particle, "z"].values

        trk = np.polyfit(z, r, 1)
        # z0 is the z-intercept, so find when r = 0
        z0 = -trk[1] / trk[0]
        z0s.append(z0)

    results = pd.DataFrame({"particle_id": list_of_particles, "z0": z0s})

    return results


def pre_process(
    list_of_hits: list, list_of_particles: list, list_of_truth: list
) -> pd.DataFrame:

    all_tracks = pd.DataFrame({})
    event_number = 0
    for f_hits, f_particles, f_truth in zip(
        list_of_hits, list_of_particles, list_of_truth
    ):

        pos = f_hits.find("event")
        if f_hits[pos + 5 : pos + 14] != f_particles[pos + 5 : pos + 14]:
            logging.error("Event number mismatch")
        if f_hits[pos + 5 : pos + 14] != f_truth[pos + 5 : pos + 14]:
            logging.error("Event number mismatch")
        if f_particles[pos + 5 : pos + 14] != f_truth[pos + 5 : pos + 14]:
            logging.error("Event number mismatch")

        logging.info(f"Running file: {f_hits[pos+5:pos+14]}")
        hits = pd.read_csv(f_hits)
        particles = pd.read_csv(f_particles)
        truth = pd.read_csv(f_truth)

        hits["r"] = calc_r(hits)
        hits["x"] = hits["x"] / 10
        hits["y"] = hits["y"] / 10
        hits["z"] = hits["z"] / 10
        hits["eta"] = calc_eta(hits["r"], hits["z"])

        particles["pt"] = calc_pt(particles)

        mask_pt = particles["pt"] > 2
        mask_nhits = particles["nhits"] > 4
        particles = particles.loc[mask_pt & mask_nhits]

        truth = truth.loc[
            truth["particle_id"].isin(
                particles.loc[mask_pt & mask_nhits, "particle_id"]
            )
        ]
        # mask_eta = np.abs(hits["eta"] < 2.4)
        hits = hits.loc[hits.hit_id.isin(truth.hit_id)]
        hits = pd.merge(
            hits, truth[["hit_id", "particle_id"]], how="left", on=["hit_id"]
        )

        trk_z0 = calc_z0(hits)
        trk_z0 = pd.merge(
            trk_z0, particles[["particle_id", "pt"]], on="particle_id", how="left"
        )
        trk_z0["event_number"] = event_number
        event_number += 1  # increment event number
        trk_z0.reset_index(inplace=True)
        trk_z0.rename(columns={"index": "track_number"}, inplace=True)
        all_tracks = pd.concat([all_tracks, trk_z0])

    all_tracks.to_pickle("all_tracks.pkl")
    print("tracks saved! ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run trackML pre-processing")
    parser.add_argument(
        "-f",
        "--folder",
        help="folder where files are stored",
        required=True,
        type=str,
    )
    args = vars(parser.parse_args())

    folder = args["folder"]

    list_of_cells = sorted(glob.glob(folder + "/event00000*-cells.csv"))
    list_of_hits = sorted(glob.glob(folder + "/event00000*-hits.csv"))
    list_of_particles = sorted(glob.glob(folder + "/event00000*-particles.csv"))
    list_of_truth = sorted(glob.glob(folder + "/event00000*-truth.csv"))

    print(f"processing a total of {len(list_of_cells)} files")

    pre_process(list_of_hits, list_of_particles, list_of_truth)
