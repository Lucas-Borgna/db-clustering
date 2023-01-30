import pandas as pd
import numpy as np
import math
import itertools
import copy


class BatchedDBSCAN:
    def __init__(
        self,
        z0,
        pt,
        eps,
        batch_size,
        max_number_of_tracks,
        top_pt_n: int = 10,
        fh_nbins: int = 5,
    ):

        # predefined constants
        self.z0_boundary = 21
        self.pt_boundary = 0
        self.minPts = 2

        self.eps = eps
        self.batch_size = batch_size
        self.max_number_of_tracks = int(max_number_of_tracks)
        self.n_batches = math.ceil(self.max_number_of_tracks / self.batch_size)
        self.top_pt_n = top_pt_n
        self.fh_nbins = fh_nbins

        # Max number of tracks including all batches
        self.max_n_tracks_batched = self.batch_size * self.n_batches
        self.max_n_clusters_batch = math.ceil(self.batch_size / self.minPts)  #
        self.max_n_clusters = math.ceil(self.max_n_tracks_batched / self.minPts)

        # need to pad vectors to have the fixed size
        n_pad = self.max_number_of_tracks - z0.shape[0]
        self.z0 = self.pad_vector(z0, n_pad, self.z0_boundary)
        self.pt = self.pad_vector(pt, n_pad, self.pt_boundary)

        # Prefix sum variables
        self.max_number_of_tracks_power_2 = (
            1 << (self.max_number_of_tracks - 1).bit_length()
        )
        self.batch_size_power_2 = 1 << (self.batch_size - 1).bit_length()
        self.max_number_of_tracks_log_2 = np.log2(self.max_number_of_tracks_power_2)
        self.batch_size_log_2 = np.log2(self.batch_size_power_2)

        # Dictionaries for recording the results
        self.results = {}
        self.results_sklearn = {}
        self.merged_list = []

        # Cluster array indices
        # pT, z0_low, z0_high, Noise
        self.pt_idx = 0
        self.z0_low_idx = 1
        self.z0_high_idx = 2
        self.noise_idx = 3

    def pad_vector(self, vec: np.array, n_pad: int, value: int) -> np.array:
        """Pads input vector with n_pad entries that have a specific value

        Args:
            vec (np.array): input vector
            n_pad (int): number of entries to add to vector
            value (int): value to pad the vector with

        Returns:
            np.array: padded vector
        """
        vec_to_pad = value * np.ones(n_pad)

        vec = np.append(vec, vec_to_pad)

        return vec

    def build_tracks(self, z0, pt) -> np.array:
        """Builds the tracks array using the input vectors of z0 and pt
        Sorts the tracks by z0.

        Args:
            z0 (np.array): input z0 values
            pt (np.array): input pt values
        Returns:
            np.array: tracks array sorted along the z0 axis.
        """

        # Shape is determined by the size of the batch, z0, pt
        track_batch = np.zeros((z0.shape[0], 2))
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

    def find_left_boundaries(self, tracks: np.array) -> np.array:
        """
        Finds the left boundaries in the input tracks array.
        Tracks must be sorted along the z0 axis.
        returns a boolean array indicating the left boundaries.
        """

        left_boundaries = np.zeros(tracks.shape[0], dtype=bool)

        # first value is always a left boundary
        left_boundaries[0] = 1

        for i in range(1, left_boundaries.shape[0]):

            if tracks[i][0] - tracks[i - 1][0] > self.eps:
                left_boundaries[i] = 1
            else:
                left_boundaries[i] = 0

        self.left_boundaries = left_boundaries
        return left_boundaries

    def find_boundaries(
        self, left_boundaries: np.array, rs: np.array, tracks: np.array
    ) -> np.array:
        """Finds the full boundaries of the clusters and determines their
        pt_sum. Requires the left boundaries.

        Args:
            left_boundaries (np.array): input boolean array indicating if a track is a left boundary of a cluster
            rs (np.array): input boolean array containing the prefix sum of pt values
            tracks (np.array): input tracks array.

        Returns:
            np.array: boundaries of clusters, sorted by index.
        """

        max_tracks = tracks.shape[0]

        # boundaries data type:
        # [index, pt_sum(i),pt_sum(i+1), pt_sum(i+1)-pt_sum(i),z0_low,z0_high, noise]
        boundaries = np.zeros((max_tracks, 7))
        # is_noise = np.ones((max_tracks))
        # boundaries begin as noise by default
        boundaries[:, 6] = np.ones(max_tracks)

        for i in range(max_tracks - 1):

            left_edge = left_boundaries[i] and not (left_boundaries[i + 1])  # 1, 0
            right_edge = not (left_boundaries[i]) and left_boundaries[i + 1]  # 0, 1
            check_noise = (left_boundaries[i] == 1) and (left_boundaries[i + 1] == 1)

            if left_edge or right_edge:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i + 1, 0]
                boundaries[i][6] = 0
                # is_noise[i] = 0
            elif check_noise:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i, 0]
                boundaries[i][6] = 1
                # is_noise[i] = 1
            else:
                boundaries[i][0] = max_tracks
                boundaries[i][1] = 0
                boundaries[i][2] = 0
                boundaries[i][3] = 0
                boundaries[i][4] = 21
                boundaries[i][5] = 21
                boundaries[i][6] = 0
                # is_noise[i] = 0

        # Check for the last boundary

        if left_boundaries[max_tracks - 1] and boundaries[max_tracks - 1, 6] == 0:
            boundaries[max_tracks - 1][0] = max_tracks
            boundaries[max_tracks - 1][1] = 0
            boundaries[max_tracks - 1][2] = 0
            boundaries[max_tracks - 1][3] = 0
            boundaries[max_tracks - 1][4] = 21
            boundaries[max_tracks - 1][5] = 21
        elif left_boundaries[max_tracks - 1] and boundaries[max_tracks - 1, 6] == 1:

            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]
            # boundaries[max_tracks - 1][6] = 1
        else:
            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]

        # Sort boundaries by the index
        sort_idx = boundaries[:, 0].argsort()
        boundaries = boundaries[sort_idx]
        self.is_noise = boundaries[:, 6]
        return boundaries

    def initialize_clusters(self, max_n_clusters: int) -> np.array:

        clusters = np.zeros((max_n_clusters, 4))

        clusters[:, self.z0_low_idx] = 21
        clusters[:, self.z0_high_idx] = 21

        return clusters

    def convert_boundaries_to_clusters(self, boundaries: np.array):
        bound_i = 0
        cluster_j = 0
        n_boundaries = boundaries.shape[0]

        clusters = self.initialize_clusters(self.top_pt_n)

        for _ in range(n_boundaries):
            if (bound_i + 2) >= n_boundaries:
                break
            noise = boundaries[bound_i, -1]

            if noise:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i, 5]
                pt_sum = boundaries[bound_i, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise

                x = z0_low  # Since its noise, the x is the same as z0_low or z0_high
                h = pt_sum  # Since its noise, the histo is just the pt of the noise point
                self.fxs[(self.batch_number * self.batch_size) + cluster_j] = x * h
                self.Ns[(self.batch_number * self.batch_size) + cluster_j] = h

                bound_i += 1
                cluster_j += 1
            else:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i + 1, 4]
                pt_sum = boundaries[bound_i + 1, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise

                idx = boundaries[bound_i, 0]
                idx_next = boundaries[bound_i + 1, 0]
                f_batch_mask = np.zeros(n_boundaries)
                bin_edges = np.linspace(z0_low, z0_high, self.fh_nbins)
                x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                f_batch_mask[int(idx) : int(idx_next)] = 1

                h = np.histogram(
                    self.z0_batches[self.batch_number],
                    bin_edges,
                    weights=self.pt_batches[self.batch_number] * f_batch_mask,
                )[0]

                self.fxs[(self.batch_number * self.top_pt_n) + cluster_j] = np.sum(
                    x * h
                )
                self.Ns[(self.batch_number * self.top_pt_n) + cluster_j] = np.sum(h)

                bound_i += 2
                cluster_j += 1

        return clusters

    def clusters_overlap(self, ci, cj) -> bool:
        """Function determines if cluster i overlaps with cluster j.
        Function detects if the clusters overlap by checking if the inverse
        is true.

        Args:
            ci (np.array): input cluster i
            cj (np.array): input cluster j

        Returns:
            bool: TRUE if clusters overlap, FALSE otherwise
        """
        # |---- c_i -----|   |---- c_j ----|
        case1 = ci[self.z0_low_idx] - self.eps <= cj[self.z0_high_idx]
        case2 = ci[self.z0_high_idx] + self.eps >= cj[self.z0_low_idx]

        overlap = case1 and case2

        return overlap

    def merge_clusters(self, c: np.array) -> np.array:

        clusters = c.copy()

        n_clusters = clusters.shape[0]
        if self.n_batches == 1:
            self.max_pt_i = np.argmax(clusters[:, self.pt_idx])
            self.max_pt = clusters[self.max_pt_i, self.pt_idx]
            return clusters

        else:
            max_pt = 0
            max_pt_i = 0

            comb = list(itertools.combinations(range(n_clusters), 2))
            self.comb = comb

            to_merge = 9 * np.ones((n_clusters, n_clusters))

            for i, j in comb:

                # skip if cluster  is outside detector
                if (clusters[i, self.z0_low_idx] >= 21) or (
                    clusters[j, self.z0_low_idx] >= 21
                ):
                    continue

                ci = copy.copy(clusters[i, :])
                cj = copy.copy(clusters[j, :])

                overlap = self.clusters_overlap(clusters[i, :], clusters[j, :])
                to_merge[i, j] = overlap
                if overlap:

                    # If cluster j is noise, then upon merging it is no-longer noise
                    cj_noise = clusters[j, self.noise_idx]

                    if cj_noise:
                        clusters[j, self.noise_idx] = 0

                    # Expand boundaries of cluster after merging
                    if clusters[i, self.z0_low_idx] < clusters[j, self.z0_low_idx]:
                        clusters[j, self.z0_low_idx] = clusters[i, self.z0_low_idx]
                    if clusters[i, self.z0_high_idx] > clusters[j, self.z0_high_idx]:
                        clusters[j, self.z0_high_idx] = clusters[i, self.z0_high_idx]

                    # Add the pT of the cluster being merged.
                    clusters[j, self.pt_idx] += clusters[i, self.pt_idx]

                    self.fxs[j] += self.fxs[i]
                    self.Ns[j] += self.Ns[i]

                    # Erase merged cluster.
                    clusters[i, self.pt_idx] = 0
                    clusters[i, self.z0_low_idx] = 21
                    clusters[i, self.z0_high_idx] = 21
                    clusters[i, self.noise_idx] = 0

                # check if the pT_sum max is now higher
                # Need to protect against selecting a noise point as PV
                if (max_pt < clusters[j, self.pt_idx]) and (
                    clusters[j, self.noise_idx] != 1
                ):
                    max_pt = clusters[j, self.pt_idx]
                    max_pt_i = j
            self.to_merge = pd.DataFrame(to_merge)
            self.max_pt = max_pt
            self.max_pt_i = max_pt_i

            return pd.DataFrame(
                clusters, columns=["pt_sum", "z0_low", "z0_high", "noise"]
            )

    def fit(self):

        start_idx = 0
        end_idx = start_idx + self.batch_size
        # Need to pad vectors to match the size of n_batches*batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        clusters = self.initialize_clusters(self.top_pt_n * self.n_batches)

        self.z0_batches = {}
        self.pt_batches = {}

        self.fxs = np.zeros(self.max_n_tracks_batched)
        self.Ns = np.zeros(self.max_n_tracks_batched)

        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            self.batch_number = i

            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            idx_sort = pt_batch.argsort()[::-1]
            z0_batch = z0_batch[idx_sort]
            pt_batch = pt_batch[idx_sort]
            z0_batch = z0_batch[: self.top_pt_n]
            pt_batch = pt_batch[: self.top_pt_n]

            track_batch = self.build_tracks(z0_batch, pt_batch)
            self.tracks = track_batch

            rs_batch = self.pad_vector(
                track_batch[:, 1], self.batch_size_power_2 - track_batch.shape[0], 0
            )

            rs_batch = self.prefix_sum(rs_batch)
            self.rs = rs_batch

            # Storing batches
            self.z0_batches[i] = track_batch[:, 0]
            self.pt_batches[i] = track_batch[:, 1]

            # Finding Left Boundaries
            left_boundaries = self.find_left_boundaries(track_batch)

            # Finding Right Boundaries
            boundaries = self.find_boundaries(left_boundaries, rs_batch, track_batch)

            self.boundaries = boundaries
            clusters_batch = self.convert_boundaries_to_clusters(boundaries)

            clusters[i * self.top_pt_n : (i + 1) * self.top_pt_n, :] = clusters_batch

            self.results[i] = clusters_batch

        clusters = self.merge_clusters(clusters)

        self.clusters = clusters

        self.z0_pv_wm = self.fxs[self.max_pt_i] / self.Ns[self.max_pt_i]
