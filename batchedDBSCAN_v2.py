import numpy as np
import pandas as pd
import math

# Merging of clusters
# https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap/325964#325964


class BatchedDBSCANV2:
    def __init__(
        self, z0, pt, eps, batch_size, max_number_of_tracks, verbose: bool = False
    ):

        self.eps = eps
        self.batch_size = batch_size
        self.verbose = verbose
        self.z0_boundary = 21  # 21 cm is outside the detector acceptance
        self.pt_boundary = 0  # 0 pT won't contribute to the pT sum.
        self.minPts = 2  # This algorithm only works for a minimum number of 2 points

        self.max_number_of_tracks = int(max_number_of_tracks)
        self.n_batches = math.ceil(self.max_number_of_tracks / self.batch_size)

        # Max number of tracks including all batches
        self.max_n_tracks_batched = self.batch_size * self.n_batches
        self.max_n_clusters_batch = math.ceil(self.batch_size / self.minPts)
        self.max_n_clusters = math.ceil(self.max_n_tracks_batched / self.minPts)

        # Need to pad vectors to the max_number_of_tracks allowed so that it matches the fpga input
        n_pad = self.max_number_of_tracks - z0.shape[0]
        # if verbose:
        # print("original number of tracks: ", z0.shape)
        self.z0 = self.pad_vector(z0, n_pad, self.z0_boundary)
        self.pt = self.pad_vector(pt, n_pad, self.pt_boundary)

        # These are needed for the prefix sum
        self.max_number_of_tracks_power_2 = (
            1 << (self.max_number_of_tracks - 1).bit_length()
        )
        self.batch_size_power_2 = 1 << (self.batch_size - 1).bit_length()
        self.max_number_of_tracks_log_2 = np.log2(self.max_number_of_tracks_power_2)
        self.batch_size_log_2 = np.log2(self.batch_size_power_2)
        # self.n_batches = math.ceil(self.max_number_of_tracks / self.batch_size)

    def pad_vector(self, vec, n_pad, value):
        """pads vector to a set size with given value"""

        vec_to_pad = value * np.ones(n_pad)
        vec = np.append(vec, vec_to_pad)

        return vec

    def build_tracks(self, z0, pt):
        """Builds tracks batchess"""

        # Shape is determined by the size of batch, z0, pT and label (not used atm)
        track_batch = np.zeros((self.batch_size, 3))

        track_batch[:, 0] = z0
        track_batch[:, 1] = pt

        # sort the tracks by z0
        track_batch = track_batch[track_batch[:, 0].argsort()]

        return track_batch

    def prefix_sum(self, arr):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        size_log2 = int(np.log2(arr.shape[0]))

        # up-sweep
        for d in range(0, size_log2, 1):
            step_size = 2**d
            double_step_size = step_size * 2

            for i in range(0, arr.shape[0], double_step_size):
                arr[i + double_step_size - 1] += arr[i + step_size - 1]

        # down-sweep
        arr[arr.shape[0] - 1] = 0
        d = size_log2 - 1

        while d >= 0:
            step_size = 2**d
            double_step_size = step_size * 2
            for i in range(0, arr.shape[0], double_step_size):
                tmp = arr[i + step_size - 1]
                arr[i + step_size - 1] = arr[i + double_step_size - 1]
                arr[i + double_step_size - 1] += tmp
            d -= 1

        return arr

    def find_left_boundaries(self, tracks):

        left_boundaries = np.zeros(self.batch_size, dtype=bool)

        # first value is always a left boundary
        left_boundaries[0] = 1

        for i in range(1, self.batch_size):
            _t = tracks[i]

            if _t[0] - tracks[i - 1][0] > self.eps:
                tracks[i][2] = -1
                left_boundaries[i] = 1
            else:
                left_boundaries[i] = 0

        return left_boundaries

    def find_right_boundaries(self, left_boundaries, rs, tracks):

        max_tracks = self.batch_size

        boundaries = np.zeros((max_tracks, 6))

        for i in range(max_tracks - 1):

            check1 = left_boundaries[i] and not (left_boundaries[i + 1])
            check2 = not (left_boundaries[i]) and left_boundaries[i + 1]

            if check1 or check2:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i + 1, 0]
            else:
                boundaries[i][0] = max_tracks
                boundaries[i][1] = 0
                boundaries[i][2] = 0
                boundaries[i][3] = 0
                boundaries[i][4] = 21
                boundaries[i][5] = 21

        # Check for the last boundary
        if left_boundaries[max_tracks - 1]:
            boundaries[max_tracks - 1][0] = max_tracks
            boundaries[max_tracks - 1][1] = 0
            boundaries[max_tracks - 1][2] = 0
            boundaries[max_tracks - 1][3] = 0
            boundaries[max_tracks - 1][4] = 21
            boundaries[max_tracks - 1][5] = 21
        else:
            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]

        # Sort boundaries by the index
        boundaries = boundaries[boundaries[:, 0].argsort()]

        # # Add sum of Pt information
        # boundaries[:, 3] = boundaries[:, 2] - boundaries[:,  1]

        return boundaries

    def convert_boundaries_to_clusters(self, boundaries: np.array) -> np.array:
        n_boundaries = boundaries.shape[0]
        n_clusters = math.ceil(n_boundaries / 2)
        clusters = np.zeros((n_clusters, 6))
        j = 0
        for i in range(0, n_boundaries, 2):
            pt_low = boundaries[i, 1]
            pt_high = boundaries[i + 1, 2]
            pt_sum = pt_high - pt_low
            z0_low = boundaries[i, 4]
            z0_high = boundaries[i + 1, 5]

            clusters[j, 3] = pt_sum
            clusters[j, 4] = z0_low
            clusters[j, 5] = z0_high
            j += 1
        return clusters

    def get_vertex(self, cluster_of_tracks: np.array) -> float:
        """
        Calculates the median z0 of the cluster of tracks
        """

        n_size = cluster_of_tracks.shape[0]

        if n_size % 2 == 0:
            return 0.5 * (
                cluster_of_tracks[n_size // 2] + cluster_of_tracks[n_size // 2 - 1]
            )
        else:
            return cluster_of_tracks[n_size // 2]

    def fit(self):

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

        start_idx = 0
        end_idx = start_idx + self.batch_size
        # Need to pad vectors to match the size of n_batches*batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        max_rs = 0

        clusters = np.zeros((self.max_n_clusters, 6))

        self.max_pt = 0
        self.max_pt_i = 0
        self.merge_count = 0

        pv_cluster = np.zeros((1, 6))

        for i in range(self.n_batches):

            # print(f"----------------- {i} --------------")
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size

            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            track_batch = self.build_tracks(z0_batch, pt_batch)

            rs_batch = self.pad_vector(
                track_batch[:, 1], self.batch_size_power_2 - self.batch_size, 0
            )

            rs_batch = self.prefix_sum(rs_batch)

            left_boundaries = self.find_left_boundaries(track_batch)

            boundaries = self.find_right_boundaries(
                left_boundaries, rs_batch, track_batch
            )
            clusters_batch = self.convert_boundaries_to_clusters(boundaries)

            # Add batch to current list of clusters
            clusters[
                i * self.max_n_clusters_batch : (i + 1) * self.max_n_clusters_batch,
                :,
            ] = clusters_batch

            # Perform the merging of the clusters with the previously found clusters
            if i >= 1:
                clusters = self.merge_clusters(clusters)

            # Stop if we've reached the maximum
            if track_batch[-1, 0] == 21:
                break

        # Find pv_cluster
        pv_cluster[0, :] = clusters[self.max_pt_i, :]
        # print("---------clusters---------")
        # print(clusters)
        # np.save("clusters.npy", clusters)
        print(self.max_pt, self.max_pt_i)
        # print(pv_cluster)
        print(f"Merged count: {self.merge_count}")

        pv_tracks = []

        for i in range(self.max_number_of_tracks):
            z0_trk = self.z0[i]

            if (z0_trk >= pv_cluster[0, 4]) and (z0_trk <= pv_cluster[0, 5]):
                pv_tracks.append(z0_trk)

        median_vertex = self.get_vertex(np.array(pv_tracks))

        print(f"mean: {np.mean(pv_tracks)}")
        print(f"median: {np.median(pv_tracks)}")
        print(f"median2: {median_vertex}")

    def merge_clusters(self, clusters: np.array) -> np.array:

        n_clusters = clusters.shape[0]
        pt_idx = 3
        z0_min = 4
        z0_max = 5

        for i in range(n_clusters):

            c_i = clusters[i, :]

            # skip if we've reached the end of the clusters
            if c_i[z0_min] >= 21:
                continue

            cluster_pt = c_i[pt_idx]
            if self.max_pt < cluster_pt:
                self.max_pt = cluster_pt
                self.max_pt_i = i

            for j in range(n_clusters):
                if i >= j:
                    continue
                c_j = clusters[j, :]

                # Skip if we've reached the end of the clusters
                if c_j[z0_min] >= 21:
                    continue

                # left_overlap (c_j, before c_i)
                left_overlap = (c_i[z0_min] - self.eps) <= c_j[z0_max]

                # right_overlap (c_i befrore c_j)
                right_overlap = (c_i[z0_max] + self.eps) >= c_j[z0_min]

                # Merge if both conditions are true
                merge = left_overlap and right_overlap

                if merge:
                    self.merge_count += 1

                    # Expand cluster boundaries if necessary
                    if c_j[z0_min] < c_i[z0_min]:
                        clusters[i, z0_min] = c_j[z0_min]
                    if c_j[z0_max] > c_i[z0_max]:
                        clusters[i, z0_max] = c_j[z0_max]
                    # Add pT to existing cluster
                    clusters[i, pt_idx] += clusters[j, pt_idx]

                    # Check if max_pt is greater than before
                    if self.max_pt < clusters[i, pt_idx]:
                        self.max_pt = clusters[i, pt_idx]
                        self.max_pt_i = i

                    # Delete cluster post merging
                    clusters[j, pt_idx] = 0
                    clusters[j, z0_min] = 21
                    clusters[j, z0_max] = 21
            return clusters
