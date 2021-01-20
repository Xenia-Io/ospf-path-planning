""" Implementation of OSPF protocol

    For the Degree Project at KTH Royal Institute
    of Technology - May 2020
"""

__author__ = "Xenia Ioannidou"

import networkx as nx
import matplotlib.pyplot as plt
import collections


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, node_name=None):
        self.parent = parent
        self.node_name = node_name

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.node_name == other.node_name


class Ospf():
    """
       OSPF class
    """

    def __init__(self):
        self.G = nx.Graph()
        self.nodes = {'rt1':{'rt5':0.005, 'rt6':0.005, 'rt7':0.005, 'rt8':0.005},
                      'rt2': {'rt5': 0.005, 'rt6': 0.005, 'rt7': 0.005, 'rt8': 0.005},
                      'rt3': {'rt5': 0.005, 'rt6': 0.005, 'rt7': 0.005, 'rt8': 0.005},
                      'rt4': {'rt5': 0.005, 'rt6': 0.005, 'rt7': 0.005, 'rt8': 0.005},
                      'rt5': {'rt9': 0.005, 'rt10': 0.005, 'rt1': 0.005, 'rt2':0.005, 'rt3':0.005, 'rt4':0.005},
                      'rt6': {'rt11':0.005, 'rt12': 0.005, 'rt1': 0.005, 'rt2':0.005, 'rt3':0.005, 'rt4':0.005},
                      'rt7': {'rt13':0.005, 'rt14': 0.005, 'rt1': 0.005, 'rt2':0.005, 'rt3':0.005, 'rt4':0.005},
                      'rt8': {'rt15':0.005, 'rt16': 0.005, 'rt1': 0.005, 'rt2':0.005, 'rt3':0.005, 'rt4':0.005},
                      'rt9': {'rt17':0.01, 'rt5': 0.005}, 'rt10': {'rt17':0.01, 'rt5': 0.005},
                      'rt11': {'rt18':0.01, 'rt6': 0.005}, 'rt12':{'rt18':0.01, 'rt6': 0.005},
                      'rt13': {'rt19':0.01, 'rt7': 0.005}, 'rt14': {'rt19': 0.01, 'rt7': 0.005},
                      'rt15': {'rt20':0.01, 'rt8': 0.005}, 'rt16':{'rt20': 0.01, 'rt8': 0.005},
                      'rt17': {'rt21': 0.01, 'rt25':0.01, 'rt9': 0.01, 'rt10': 0.01},
                      'rt18': {'rt22': 0.01, 'rt25': 0.01, 'rt11': 0.01, 'rt12': 0.01},
                      'rt19': {'rt23': 0.01, 'rt25': 0.01, 'rt13': 0.01, 'rt14': 0.01},
                      'rt20': {'rt24': 0.01, 'rt25': 0.01, 'rt15': 0.01, 'rt16': 0.01},
                      'rt21': {'rt17': 0.01}, 'rt22':{'rt18':0.01},'rt23':{'rt19':0.01}, 'rt24':{'rt20':0.01},
                      'rt25': {'rt26': 0.01, 'rt27': 0.01, 'rt17': 0.01, 'rt18': 0.01, 'rt19': 0.01, 'rt20': 0.01},
                      'rt26': {'rt25': 0.01}, 'rt27': {'rt25': 0.01}}
        self.commands = ['add rt', 'add nt', 'del rt', 'del nt', 'con', 'tree', 'display']
        self.KILL = False
        self.answer_map = {}


    def add_edge(self, start, end, weight):
        self.G.add_weighted_edges_from([(start, end, weight)])
        # self.G.add_edge(*e)


    def add_edge_list(self,edgelist):
        for i in range(len(edgelist)):
            self.G.add_edge(*edgelist[i])


    def add_new_edge(self, cnx):

        # check that cnx[3], aka, connection cost is integer
        try:
            abs_val = (float(cnx[2]) - int(cnx[2]))
        except ValueError:
            print("ERROR: Edge cost must be an integer")
            return

        # check that argument list is length 3
        if len(cnx) != 3:
            print("ERROR: Must enter 3 arguments")
            return
        # check that user is not connecting node to itself
        elif cnx[0] == cnx[1]:
            print("ERROR: Cannot connect a node to itself")
            return
        # check that user is not connecting two networks
        elif cnx[0][:2] == "nt" and cnx[1][:2] == "nt":
            print("ERROR: Cannot connect two networks")
            return
        # check that both nodes exist in the network
        elif cnx[0] not in self.nodes:
            print("ERROR: " + cnx[0] + " does not exist in network")
            return
        elif cnx[1] not in self.nodes:
            print("ERROR: " + cnx[1] + " does not exist in network")
            return
        # check that connection is integer type
        elif abs_val != 0:
            print("ERROR: Edge cost must be an integer")
            return
        # check that connection cost is >= 1
        elif int(cnx[2]) < 1:
            print("ERROR: Edge cost must be greater than or equal to 1")
            return

        self.nodes[cnx[0]][cnx[1]] = int(cnx[2])
        print("Connection from " + cnx[0] + " to " + cnx[1] + " made successfully")
        return

    def remove_connection(self, cnx):
        print( cnx, type(cnx), len(cnx))
        params = cnx[0].split(' ')
        print(type(params[0]), params[0], params[1])
        del self.nodes[params[0]][params[1]]
        print("Connection from " + params[0] + " to " + params[1] + " removed successfully")
        return

    def plot_graph(self):
        pos = nx.spring_layout(self.G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=300)

        # edges
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(data=True),
                               width=2)
        # labels
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family='sans-serif')

        plt.axis('off')
        plt.show()


    def create_weighed_graph(self, edges_dict):
        if self.G.nodes:
            self.G.clear()
        print("Start building network with edges: ", edges_dict)
        for key in edges_dict:
            # print("key = ", key, " with value = ", edges_dict[key])
            key_split = str(key).replace("(","").replace(")","").split(",")
            self.G.add_edge(key_split[0], key_split[1], weight=edges_dict[key])


    def add_node(self, rt):
        node = 'rt' + str(rt)
        if node in self.nodes.keys():
            print("ERROR: Router, ", node, " already exists.")
            return
        print("Router,", node, "successfully added.")
        self.nodes[node] = {}  # adds router to nodes, no connections


    def add_node_to_graph(self, node):
        self.G.add_node(node)


    def add_nodes_to_graph(self, my_list):
        self.G.add_nodes_from(my_list)


    def remove_node_from_graph(self, node):
        self.G.remove_node(node)

    def remove_edge(self, node_from, node_to):
        e = (node_from, node_to)
        self.G.remove_edge(*e)

    def del_router(self, rt):
        node = 'rt' + str(rt)
        if node in self.nodes:
            del self.nodes[node]  # delete the main key
            for x in self.nodes:  # loop through remaining main keys in dict
                if node in self.nodes[x]:  # true if node exists as sub-node in main key 'x'
                    del self.nodes[x][node]  # delete the sub key in 'x'
            print("Router ", node, "successfully removed.")
            return
        print("ERROR: Router ", node, " does not exist.")


    def display(self):
        # print
        keys = []  # store the keys from dictionary in a list for easier display
        for key in self.nodes:
            keys.append(key)
        top_row = "   "
        row = [" "] * len(keys)  # create array of rows for link-state database
        sorted(keys)  # sorts the key list for easier reading
        i = 0
        while i < len(keys):
            top_row += "   " + keys[i]  # create string of all keys
            i += 1
        print(top_row)

        i = 0
        # The following loop creates the rows of the LSDB, and inserts the edge costs in the table
        while i < len(keys):
            row[i] = keys[i]
            j = 0
            while j < len(keys):
                row[i] += "  "
                if keys[i] in self.nodes[keys[j]]:  # The following 4 lines provide padding depending on
                    # the length of the connection(1 or 2 digits), this
                    # helps keep the display table formatted and legible
                    if len(str(self.nodes[keys[j]][keys[i]])) == 1:
                        row[i] += str(self.nodes[keys[j]][keys[i]]) + "  "
                    elif len(str(self.nodes[keys[j]][keys[i]])) == 2:
                        row[i] += str(self.nodes[keys[j]][keys[i]]) + " "
                    elif len(str(self.nodes[keys[j]][keys[i]])) == 3:
                        row[i] += str(self.nodes[keys[j]][keys[i]]) + ""
                else:
                    row[i] += "  "
                row[i] += "  "
                j += 1
            print(row[i])

            i += 1
        return

    def modified_Dijkstra(self, G, start):
        """The Dijkstra algorithm

        :param G: a dictionary about all nodes and their edges
        :param start: the source node
        :return:
        """

        final_dist = self.nodes.fromkeys(self.nodes)  # dict of final distances
        final_rout = [0] * len(G.keys())  # list of final routes
        final_cost = self.nodes.fromkeys(self.nodes)
        keys = []
        for key in G:
            keys.append(key)

        # Set all final distances to -1
        for x in final_dist:
            final_dist[x] = -1
            final_cost[x] = -1
        unvisited_nodes = self.nodes.fromkeys(self.nodes)

        final_dist[start] = 0  # set distance of source node to zero
        final_cost[start] = 0
        current = start  # start at current node

        # final_rout is a 2d array, each node has its own array to store the shortest path
        i = 0
        while i < len(keys):
            final_rout[i] = []
            i += 1

        while unvisited_nodes != {}:
            if final_dist[current] != -1:  # if node has not been relaxed
                indx_curr_node = keys.index(current)  # get index of current node
                for y in G[current]:  # for each node attached to the node currently being relaxed
                    index = keys.index(y)  # get index of adjacent node to current node currently being checked
                    if final_dist[y] == -1:  # if node has not been visited
                        final_dist[y] = (final_dist[current] + G[current][y])
                        final_cost[y] = float(final_cost[current]) + round(1/(G[current][y]),1)
                        final_rout[index] = final_rout[indx_curr_node] + [current]
                    elif (final_dist[current] + (G[current][y])) < final_dist[y] and current != start:
                        final_dist[y] = (final_dist[current] + G[current][y])
                        final_cost[y] = float(final_cost[current]) + round(1/(G[current][y]),1)
                        final_rout[index] = final_rout[indx_curr_node] + [current]
            del unvisited_nodes[current]  # delete current node from unvisited set once it has been relaxed

            # this method chooses the closest node to the current node, the closest
            # node will be relaxed next in accordance with Dijkstra Algorithm
            e_vert = []  # list will hold adjacent vertices
            for x in G:
                if x in unvisited_nodes and final_dist[x] > 0:  # all unrelaxed nodes adjecent to current node
                    e_vert.append(x)
            try:
                close_tuple = min([(final_dist[x], x) for x in e_vert])
            except ValueError:
                break
            current = close_tuple[1]  # set new current node

        return (final_dist, final_rout, keys, final_cost)


    def astar(self, network, start, end, dict):
        """Returns a list of tuples as a path from the given start to the given end in the given graph"""

        print("A* just started ...")
        print("Initial node: ", start)
        print("Target node: ", end)
        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []
        final_cost = 0.0
        open_list.append(start_node)

        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            # print("New while with this open_list: ", len(open_list))
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index


            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(int(current.node_name))
                    current = current.parent

                # Calculate the path cost
                for i in range(len(path)-1):
                    key = "("+ str(path[i]) + "," + str(path[i+1]) + ")"
                    if key in dict:
                        final_cost += round(1/float(dict[key]),0)
                    else:
                        key = "(" + str(path[i+1]) + "," + str(path[i]) + ")"
                        final_cost += round(1/float(dict[key]),0)

                return path[::-1], final_cost  # Return reversed path and final path cost

            # Get neighbors
            neighbors = []
            neighbors_edges = network.edges([str(current_node.node_name)])
            # print("z_rand: ", str(current_node.node_name), " with edges: ", neighbors_edges)
            for i in neighbors_edges:
                node_1, node_2 = i
                neighbors.append(Node(current_node, int(node_2)))


            # Sorting neighbors in ascending order regarding the weights
            dlist = network.edges([str(current_node.node_name)])
            sorted_neighbors = {}

            for i in dlist:
                sorted_neighbors[i] = float(network.get_edge_data(*i)['weight'])
            sorted_neighbors = {k: v for k, v in sorted(sorted_neighbors.items(), key=lambda item: item[1], reverse=True)}


            sorted_neighbors_list = []
            for key, val in sorted_neighbors.items():
                node_1, node_2 = key
                sorted_neighbors_list.append(Node(current_node, int(node_2)))


            # Loop through neighbors
            for neighbor in sorted_neighbors_list:

                # Neighbor is on the closed list
                for closed_neighbor in closed_list:
                    if neighbor == closed_neighbor:
                        continue

                # Create the f, g, and h values
                neighbor.g = current_node.g + 1

                # Get the edge's weight from the dictionary
                # my_key = (str(current_node.node_name), str(neighbor.node_name))
                # print("current_node.node_name: ",current_node.node_name," neighbor.node_name: ",neighbor.node_name )
                # print("debug 00 : ", sorted_neighbors[my_key])
                # neighbor.h = ((sorted_neighbors[my_key] - end_node.h) ** 2)
                neighbor.h = ((neighbor.h - current_node.h) ** 2)
                # neighbor.h = ((neighbor.node_name - end_node.node_name) ** 2)
                neighbor.f = neighbor.g + neighbor.h

                # Child is already in the open list
                for open_node in open_list:
                    if neighbor == open_node and neighbor.g < open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(neighbor)

    def bfs(self, graph, source, target, dict):
        """Return the shortest path from the source to the target in the graph"""

        source = str(source)
        target = str(target)
        parents = {source: None}
        queue = collections.deque([source])
        print("source: ", source, " with target: ", target, type(collections.deque([source])))

        # loop until queue is non-empty: exit when queue is empty
        while queue:
            node = queue.popleft()
            for neighbor in graph[str(node)]:
                if neighbor not in parents:
                    parents[str(neighbor)] = node
                    queue.append(neighbor)
                    if node == target:
                        break

        path = [target]
        while parents[str(target)] is not None:
            path.insert(0, parents[str(target)])
            target = parents[str(target)]

        # Calculate the path cost
        final_cost = 0
        for i in range(len(path) - 1):
            key = "(" + str(path[i]) + "," + str(path[i + 1]) + ")"
            if key in dict:
                final_cost += round(1 / float(dict[key]), 0)
            else:
                key = "(" + str(path[i + 1]) + "," + str(path[i]) + ")"
                final_cost += round(1 / float(dict[key]), 0)

        return path, final_cost


    def path_planning(self, node, dict, world, init, goal, network, daq_net):
        if node == "":
            print("Please enter a source node.")
            return
        elif node not in self.nodes:
            print("ERROR: Router " + node + " does not exist in network")
            return

        print("Starting Dijkstra")
        D, R, keys, final_cost = self.modified_Dijkstra(self.nodes, node)
        print("final_dist: ", D)

        i = 0
        while i < (len(R)):
            j = 0
            if D[keys[i]] == 0:
                pass  # if node is source node,do not print
            elif D[keys[i]] == -1:
                route = "  : no path to " + keys[i]
                print("\t", route)
            else:
                route = str(D[keys[i]]) + " : "
                route2 = str(final_cost[keys[i]]) + " ::: "
                while j < len(R[i]):
                    route += (R[i][j]) + ", "
                    route2 += (R[i][j]) + ", "
                    j += 1
                route += keys[i]
                route2 += keys[i]
                print("\t", route)
                print("\t", route2)

            i += 1

        print("Starting BFS")
        path_bfs, cost_bfs = self.bfs(daq_net, init, goal, dict)
        print("BFS path: ", path_bfs, cost_bfs)

        print("Starting A*")
        path, cost = self.astar(network, init, goal, dict)
        print("A* path: ", path, " with cost: ", cost)
        return
