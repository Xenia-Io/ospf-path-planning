""" Implementation of OSPF protocol

    For the Degree Project at KTH Royal Institute
    of Technology - May 2020
"""
from click._compat import raw_input

__author__ = "Xenia Ioannidou"

from helpers import read_file
from ospf_protocol import *


class Tests():

    def __init__(self):
        self.world = []
        self.dict = {}

    def initialize_vars(self, fname):
        content_array = read_file(fname)
        init = 0
        goal = 0

        for line in range(len(content_array)):
            array = content_array[line].split("-")

            if array[0] == 'w':
                self.world = [int(a) for a in array[1:]]

            if array[0] == 'i':
                start_node_number = int(array[1])
                init = start_node_number

            if array[0] == 'g':
                goal_node_name = int(array[1])
                goal = goal_node_name

            if array[0] == 'e':
                for i in array[1:]:
                    list = (i.split(":"))
                    self.dict.update({list[0]: list[1]})

        return self.dict, self.world, init, goal


    def check_cmd(self, cmd):
        if cmd == "":
            return False

        elif cmd[0] == 'a':
            if cmd[:7] == 'add rt ':
                params = cmd[7:].split(',')
                return 'add rt', params

            elif cmd[:7] == 'add nt ':
                params = cmd[7:].split(',')
                return 'add nt', params

        elif cmd[0] == 'd':
            if cmd[:7] == 'del rt ':
                params = cmd[7:].split(',')
                return 'del rt', params

            elif cmd[:7] == 'del nt ':
                params = cmd[7:].split(',')
                return 'del nt', params

            elif cmd[:7] == 'display' and cmd[7:] == "":
                return 'display', []

        elif cmd[:4] == 'con ':
            params = cmd[4:].split(' ')
            return 'con', params

        elif cmd[:5] == 'tree ':
            params = cmd[5:]
            return 'tree', params

        elif cmd[:4] == 'quit' and cmd[4:] == "":
            return 'quit'

        return False

    def create_weighed_graph(self, network, edges_dict):
        print("Start building network with edges: ", edges_dict)
        for key in edges_dict:
            print("key = ", key, " with value = ", edges_dict[key])
            key_split = str(key).replace("(","").replace(")","").split(",")
            network.add_edge(key_split[0], key_split[1], weight=edges_dict[key])
        return network


    def build_graph_representation(self, ospf_obj):
        ospf_obj.create_weighed_graph(self.dict)
        ospf_obj.plot_graph()


    def run(self):
        show_graph = False
        ospf_obj = Ospf()
        self.dict, self.world, self.init, self.goal = self.initialize_vars("input.txt")
        self.daq_net = {
            "1": ["5", "6", "7", "8"],
            "2": ["5", "6", "7", "8"],
            "3": ["5", "6", "7", "8"],
            "4": ["5", "6", "7", "8"],
            "5": ["1", "2", "3", "4", "9", "10"],
            "6": ["1", "2", "3", "4", "11", "12"],
            "7": ["1", "2", "3", "4", "13", "14"],
            "8": ["1", "2", "3", "4", "15", "16"],
            "9": ["5", "17"],
            "10": ["5", "17"],
            "11": ["6", "18"],
            "12": ["6", "18"],
            "13": ["7", "19"],
            "14": ["7", "19"],
            "15": ["8", "20"],
            "16": ["8", "20"],
            "17": ["9", "10", "21", "25"],
            "18": ["11", "12", "25", "22"],
            "19": ["13", "14", "25", "23"],
            "20": ["15", "16", "25", "24"],
            "21": ["17"],
            "22": ["18"],
            "23": ["19"],
            "24": ["20"],
            "25": ["17", "18", "19", "20", "26", "27"],
            "26": ["25"],
            "27": ["25"]
        }

        while ospf_obj.KILL == False:
            u_input = raw_input("\nEnter a command...\n")
            c_in = self.check_cmd(u_input)
            if c_in == False:
                print("Command not found.")
                continue
            elif c_in == 'quit':
                ospf_obj.KILL = True
            else:
                if show_graph:
                    self.build_graph_representation(ospf_obj)

                index = ospf_obj.commands.index(str(c_in[0]))  # takes index number of the command
                nodes = c_in[1]  # nodes are the second input parameter
                if nodes == ['']:
                    print("No nodes entered")
                    continue

                # parse the router list
                if index == 0:
                    print("Adding new node")
                    for node in nodes:
                        ospf_obj.add_node(node)
                        self.world.append(int(node))
                        print("Now the network's node list has a new node: ", self.world)
                        if ospf_obj.G.nodes:
                            ospf_obj.G.clear()
                            ospf_obj.add_node_to_graph(node)
                        else:
                            ospf_obj.add_node_to_graph(node)

                elif index == 2:
                    print("Deleting a node")
                    for node in nodes:
                        ospf_obj.del_router(node)
                        print("Finaly before deleting node: ", self.world, "and nodes: ", nodes)
                        self.world.remove(int(node))
                        print("Finaly after deleting node: ", self.world)
                        ospf_obj.G.clear()
                        ospf_obj.add_nodes_to_graph(self.world)

                elif index == 3:
                    print("Deleting an edge")
                    nodes_cp = nodes
                    params = nodes[0].split(' ')
                    node_from = params[0].replace("rt", "")
                    node_to = params[1].replace("rt", "")
                    node_key = '(' + str(node_from) + ',' + str(node_to) + ')'
                    del self.dict[node_key]
                    ospf_obj.remove_edge(node_from, node_to)
                    if show_graph:
                        ospf_obj.create_weighed_graph(self.dict)
                        ospf_obj.plot_graph()

                    ospf_obj.remove_connection(nodes_cp)

                elif index == 4:
                    print("Adding new edge")
                    node_from = nodes[0]
                    node_to = nodes[1]
                    weight = nodes[2]
                    ospf_obj.add_new_edge(nodes)
                    node_from = node_from.replace("rt", "")
                    node_to = node_to.replace("rt","")
                    node_key = '(' + str(node_from)+ ',' + str(node_to) + ')'
                    self.dict.update({node_key: weight})
                    if show_graph:
                        ospf_obj.create_weighed_graph(self.dict)
                        ospf_obj.plot_graph()

                elif index == 5:
                    network = nx.Graph()
                    network = self.create_weighed_graph(network, self.dict)
                    ospf_obj.path_planning(nodes, self.dict, self.world, self.init, self.goal, network, self.daq_net)

                elif index == 6:
                    ospf_obj.display()

                else:
                    print("Method not found!")
