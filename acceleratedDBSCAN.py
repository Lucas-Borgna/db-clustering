import numpy as np
import pandas as pd
import math


class AccDBSCAN:
    def __init__(self, z0, pt, eps, max_number_of_tracks, verbose: bool = False):
        self.z0 = z0
        self.pt = pt
        self.eps = eps
        self.verbose = verbose
        self.n_tracks = z0.shape[0]
        self.rs = pt  # variable to contain the sum of pt
        self.max_number_of_tracks = int(max_number_of_tracks)
        self.max_number_of_tracks_power_2 = 1 << (max_number_of_tracks - 1).bit_length()
        self.max_number_of_tracks_log_2 = np.log2(self.max_number_of_tracks_power_2)

    def add_padding(self):
        """pads array to match the maximum number of tracks"""

        n_pad = self.max_number_of_tracks - self.n_tracks

        # 21 cm is like out of bounds of the tracker
        to_pad_z0 = 21 * np.ones(n_pad)
        self.z0 = np.append(self.z0, to_pad_z0)

        # 0 GeV padding for pT
        to_pad_pt = np.zeros(n_pad)
        self.pt = np.append(self.pt, to_pad_pt)

        # 0 GeV padding to nearesr power of 2 for prefix sum of pT
        n_pad_rs = self.max_number_of_tracks_power_2 - self.n_tracks
        to_pad_rs = np.zeros(n_pad_rs)
        self.rs = np.append(self.rs, to_pad_rs)

    def build_tracks(self):
        """
        builds tracks by putting together the [z0, pt, label] information.
        labels are initialized to 0 first
        """
        self.tracks = np.zeros((self.max_number_of_tracks, 3))

        self.tracks[:, 0] = self.z0
        self.tracks[:, 1] = self.pt

        # sort the tracks by z0
        self.tracks = self.tracks[self.tracks[:, 0].argsort()]

    def initialize_data(self):

        # pad z0 and pt
        self.add_padding()
        if self.verbose:
            print("data padded")

        # build tracks
        self.build_tracks()
        if self.verbose:
            print("tracks built")

        self.prefix_sum()
        if self.verbose:
            print("prefix sum done")

    def fit(self):

        # setup tracks and sorting
        self.initialize_data()
        if self.verbose:
            print("data initialized...")
            np.save("rs.npy", self.rs)
            np.save("pt.npy", self.pt)

        # find left boundaries
        self.find_left_boundaries()
        if self.verbose:
            print("left boundaries found...")
            np.save("left_boundaries.npy", self.left_boundaries)

        # find right boundaries
        self.find_right_boundaries()
        if self.verbose:
            print("right boundaries found...")
            np.save("right_boundaries.npy", self.boundaries)
            # np.save("full_boundaries.npy", self.boundaries)
        # find vertices
        self.find_vertices()
        if self.verbose:
            print("vertices found...")

        # record z0 of primary vertex
        self.pv_z0 = self.vertices[0][0]
        self.pv_pt = self.vertices[0][1]
        if self.verbose:
            print("scan complete.")

    def find_left_boundaries(self):

        left_boundaries = np.zeros(self.max_number_of_tracks, dtype=bool)

        # first value is always a left boundary
        left_boundaries[0] = 1

        for i in range(1, self.max_number_of_tracks):
            _t = self.tracks[i]

            if _t[0] - self.tracks[i - 1][0] > self.eps:
                self.tracks[i][2] = -1
                left_boundaries[i] = 1
            else:
                left_boundaries[i] = 0

        self.left_boundaries = left_boundaries

    def find_right_boundaries(self):

        max_tracks = self.max_number_of_tracks

        boundaries = np.zeros((max_tracks, 6))

        for i in range(max_tracks - 1):

            check1 = self.left_boundaries[i] and not (self.left_boundaries[i + 1])
            check2 = not (self.left_boundaries[i]) and self.left_boundaries[i + 1]

            if check1 or check2:
                boundaries[i][0] = i
                boundaries[i][1] = self.rs[i]
                boundaries[i][2] = self.rs[i + 1]
                boundaries[i][3] = self.rs[i + 1] - self.rs[i]
                boundaries[i][4] = self.tracks[i, 0]
                boundaries[i][5] = self.tracks[i + 1, 0]
            else:
                boundaries[i][0] = max_tracks
                boundaries[i][1] = 0
                boundaries[i][2] = 0
                boundaries[i][3] = 0
                boundaries[i][4] = 21
                boundaries[i][5] = 21

        # Check for the last boundary
        if self.left_boundaries[max_tracks - 1]:
            boundaries[max_tracks - 1][0] = max_tracks
            boundaries[max_tracks - 1][1] = 0
            boundaries[max_tracks - 1][2] = 0
            boundaries[max_tracks - 1][3] = 0
            boundaries[max_tracks - 1][4] = 21
            boundaries[max_tracks - 1][5] = 21
        else:
            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = self.rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = self.rs[max_tracks]

        # Sort boundaries by the index
        boundaries = boundaries[boundaries[:, 0].argsort()]

        self.boundaries = boundaries

    def find_vertices(self) -> np.array:
        label = 0
        max_tracks = self.max_number_of_tracks
        max_vertices = math.ceil(max_tracks / 2)

        vertices = np.zeros((max_vertices, 2))

        for i in range(0, max_tracks, 2):
            left_boundary = self.boundaries[i]
            right_boundary = self.boundaries[i + 1]

            if left_boundary[0] != right_boundary[0]:
                label += 1
                z0_vertex = self.find_vertex_and_label_clusters(
                    self.tracks, left_boundary[0], right_boundary[0], label
                )
                # print(z0_vertex)
                vertices[i // 2][0] = z0_vertex
                vertices[i // 2][1] = right_boundary[2] - left_boundary[1]

        # Argsort sorts in increasing order (add argsort[::-1][:n] for descending order)

        vertices = vertices[vertices[:, 1].argsort()[::-1][: vertices.shape[0]]]

        self.vertices = vertices

    def prefix_sum(self):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        arr = self.rs
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

    def get_vertex(self, cluster_of_tracks: np.array) -> float:
        """
        Calculates the median z0 of the cluster of tracks
        """

        n_size = cluster_of_tracks.shape[0]

        if n_size % 2 == 0:
            return 0.5 * (
                cluster_of_tracks[n_size // 2][0]
                + cluster_of_tracks[n_size // 2 - 1][0]
            )
        else:
            return cluster_of_tracks[n_size // 2][0]

    def find_vertex_and_label_clusters(
        self, tracks: np.array, startIndex: int, endIndex: int, label: int
    ) -> float:

        tracks_cluster = tracks[int(startIndex) : int(endIndex) + 1]

        z0_vertex = self.get_vertex(tracks_cluster)

        return z0_vertex
