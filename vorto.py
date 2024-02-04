import sys
import ast
import copy
import random
import numpy as np
from tqdm import tqdm

CAPACITY = 720
TEMPERATURE = 1800
ALPHA = 0.99
PENALTY = 1e7
DRIVER_COST = 500


class Load:
    def __init__(self, id, start_point, end_point):
        self.id = id            # the id of current load
        self.start = start_point
        self.end = end_point
        self.arrival_cost = dist((0, 0), start_point)
        self.return_cost = dist((0, 0), end_point)
        self.load_cost = dist(start_point, end_point)

    def __eq__(self, other):
        return self.id == other.id and self.start == other.start and self.end == other.end

    def __str__(self):
        return ("The load {} with start point {} and end point {} has the attributes\n"
                "- arrival cost is {}\n"
                "- return cost is {}\n"
                "- load cost is {}\n").format(self.id, self.start, self.end,
                                              self.arrival_cost, self.return_cost, self.load_cost)

    def __repr__(self):
        return str(self)


class Cvrp:
    def __init__(self):
        self.depot = Load(0, (0, 0), (0, 0))
        self.curr_path = []             # store the current path
        self.best_path = []             # store the best path

    def load_data(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                curr = line[:-1].split(" ")
                if curr[0].isnumeric():         # this line represents a load
                    curr_node = Load(int(curr[0]), ast.literal_eval(curr[1]), ast.literal_eval(curr[2]))
                    self.curr_path.append(curr_node)

        random.shuffle(self.curr_path)                   # initialize the path randomly
        self.curr_path = self.insert_depot()        # divide the whole path into segments for multiple drivers
        self.best_path = self.curr_path

    def insert_depot(self):
        result_path = [self.curr_path[0]]            # add the 1st load anyway
        curr_cost = self.curr_path[0].arrival_cost + self.curr_path[0].return_cost + self.curr_path[0].load_cost

        for i in range(1, len(self.curr_path)):
            # when we want to let the current driver take one more load, check
            # add the current load's load_cost + return_cost + from the previous load to current load
            # subtract the previous load's return_cost
            new_cost = curr_cost + dist(self.curr_path[i-1].end, self.curr_path[i].start) - \
                       self.curr_path[i-1].return_cost + self.curr_path[i].load_cost + self.curr_path[i].return_cost
            if new_cost > CAPACITY:
                # start a new driver for the following load
                result_path.append(self.depot)
                curr_cost = self.curr_path[i].load_cost + self.curr_path[i].arrival_cost + self.curr_path[i].return_cost
            else:
                curr_cost = new_cost
            result_path.append(self.curr_path[i])

        result_path.append(self.depot)
        result_path.insert(0, self.depot)
        return result_path

    def get_curr_cost(self, path):
        """
        Compute the total cost of the current path. self.path represents the segments of each driver.
        Total cost = 500 * drivers + total distance
        - path: the path we want to compute the cost on
        """
        num_drivers = 0
        dist_cost = 0
        segment_cost = 0

        for i in range(1, len(path)):          # ignore the first depot, not helpful
            if path[i] == self.depot:          # the end of a segment
                segment_cost += path[i - 1].return_cost
                dist_cost += segment_cost
                # if this segment is larger than capacity, ban it (add a large cost) so it will not be selected
                if segment_cost > CAPACITY:
                    dist_cost += PENALTY
                if path[i-1] != self.depot:    # SA may give two consecutive depots
                    num_drivers += 1
                segment_cost = 0
            elif path[i-1] == self.depot:      # the start of a segment, add arrival_cost
                segment_cost = path[i].arrival_cost + path[i].load_cost
            else:                                   # the middle of a segment, add load_cost and distance to here
                segment_cost += dist(path[i-1].end, path[i].start) + path[i].load_cost

        return DRIVER_COST * num_drivers + dist_cost

    def local_search(self, num_local_search):
        """
        Perform the local search from current path (self.curr_path) and decide change by SA criterion.
        If the final path is better than the current best_path, update it.

        Notice we need copy.deepcopy in each copy to avoid alias changing.

        - num_local_search: how many times we perform local search in one iteration
        """
        temp_best_path = self.curr_path
        temp_path = copy.deepcopy(self.curr_path)
        for _ in range(num_local_search):
            load1_idx, load2_idx = search_try(temp_path)
            # if self.get_curr_cost(temp_path_tried) < self.get_curr_cost(temp_best_path):
            #     temp_best_path = temp_path_tried
            #     temp_path = temp_path_tried
            # search_try(temp_path)
            if self.get_curr_cost(temp_path) < self.get_curr_cost(temp_best_path):
                temp_best_path = temp_path
            else:                           # if this change is not better, reverse it
                search_reverse(temp_path, load1_idx, load2_idx)

        # now, curr_best_path is the best path we have in this local search
        curr_cost = self.get_curr_cost(self.curr_path)
        temp_best_cost = self.get_curr_cost(temp_best_path)
        if temp_best_cost < curr_cost:                                  # update self.path
            if temp_best_cost < self.get_curr_cost(self.best_path):     # update self.best_path
                self.best_path = temp_best_path
            self.curr_path = temp_best_path
        else:
            if np.exp((curr_cost - temp_best_cost) / TEMPERATURE) > random.random():        # SA criterion
                self.curr_path = temp_best_path

        # curr_cost = self.get_curr_cost(self.curr_path)
        # temp_cost = curr_cost
        # temp_best_cost = curr_cost
        # temp_path = copy.deepcopy(self.curr_path)
        # temp_best_path = self.curr_path
        # for _ in range(num_local_search):
        #     path = temp_path
        #     idx1, idx2 = search_try(temp_path)
        #     # below is the new temp_cost after swap two loads
        #     temp_cost_new = + dist(path[idx1 - 1].end, path[idx1].start) + dist(path[idx1].end, path[idx1 + 1].start) \
        #                     + dist(path[idx2 - 1].end, path[idx2].start) + dist(path[idx2].end, path[idx2 + 1].start) \
        #                     - dist(path[idx1 - 1].end, path[idx2].start) - dist(path[idx2].end, path[idx1 + 1].start) \
        #                     - dist(path[idx2 - 1].end, path[idx1].start) - dist(path[idx1].end, path[idx2 + 1].start) \
        #                     + temp_cost
        #     if self.segment_penalty(temp_path, idx1) or self.segment_penalty(temp_path, idx2):
        #         temp_cost_new += PENALTY
        #     if temp_cost_new < temp_best_cost:
        #         temp_best_path = temp_path
        #         temp_best_cost = temp_cost_new
        #         temp_cost = temp_cost_new
        #     else:                                   # if this change is not better, reverse the change
        #         search_reverse(temp_path, idx1, idx2)
        #
        # if temp_best_cost < curr_cost:  # update self.path
        #     if temp_best_cost < self.get_curr_cost(self.best_path):  # update self.best_path
        #         self.best_path = temp_best_path
        #     self.curr_path = temp_best_path
        # else:
        #     if np.exp((curr_cost - temp_best_cost) / TEMPERATURE) > random.random():  # SA criterion
        #         self.curr_path = temp_best_path

    def display_best_path(self):
        curr_segment = []
        for i in range(1, len(self.best_path)):
            if self.best_path[i] != self.depot:
                curr_segment.append(self.best_path[i].id)
            elif self.best_path[i-1] != self.depot:
                print(curr_segment)
                curr_segment = []

    def segment_penalty(self, path, index):
        total_cost = path[index].load_cost
        left, right = index, index
        while left >= 1 and path[left - 1] != self.depot:
            total_cost += path[left].load_cost + dist(path[left].end, path[left + 1].start)
            left -= 1
        total_cost += path[left].arrival_cost
        while right <= len(path) - 2 and path[right + 1] != self.depot:
            total_cost += path[right].load_cost + dist(path[right].start, path[right - 1].end)
            right += 1
        total_cost += path[right].return_cost
        return total_cost > CAPACITY


def dist(start_point, end_point):
    return np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)


def search_try(path):
    """
    Randomly try to exchange two loads in the current path
    Notice that we may exchange depot, but ignore the first and last depot as they are not part of a path.
    - path: a list of loads representing the current path
    """
    # result_path = copy.deepcopy(path)
    # randomly pick exchanged load, except for boundary ones
    load1_idx = random.randint(1, len(path) - 2)
    load2_idx = random.randint(1, len(path) - 2)
    path[load1_idx], path[load2_idx] = path[load2_idx], path[load1_idx]
    # result_path[load1_idx], result_path[load2_idx] = result_path[load2_idx], result_path[load1_idx]
    return load1_idx, load2_idx


def search_reverse(path, index1, index2):
    path[index1], path[index2] = path[index2], path[index1]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("The input format is not correct!")
        exit(1)

    cvrp = Cvrp()
    filename = str(sys.argv[1])
    cvrp.load_data(filename)

    # costs = []
    num_iterations = 2000
    num_local_iterations = 10
    for _ in tqdm(range(num_iterations)):
        cvrp.local_search(num_local_iterations)
        TEMPERATURE *= ALPHA
    cvrp.display_best_path()

