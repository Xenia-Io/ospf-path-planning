""" Implementation of OSPF protocol

    For the Degree Project at KTH Royal Institute
    of Technology - May 2020
"""

__author__ = "Xenia Ioannidou"


def read_file(fname):
    content_array = []
    with open(fname) as f:
        # Content_list is the list that contains the read lines
        for line in f:
            content_array.append(line)
    return content_array

def writing_to_file(fname, nodes, costs):
    with open(fname, 'a') as fl:
        w = ''
        print("Start writing in file. Length of costs: ", len(costs))
        for i in range(len(costs)):
            w += str(nodes[i]) + '\t' + str(costs[i]) + '\n'
        fl.write(w)
