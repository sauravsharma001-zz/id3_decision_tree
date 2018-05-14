import sys
import pandas as pd
import numpy as np
import random
import operator

rank = 0
node_ranking = {}


class Tree(object):
    def __init__(self):
        self.child = [None] * 2
        self.accuracy = []
        self.data = None
        self.rank = -1
        self.is_visited = False


def get_count_attrib(data):
    """
    returns the count for distinct value for the attribute
    :param data: 1-D array of values for an attribute
    :return: total number of unique value for the attribute in an array representation
    """
    return np.array(np.unique(data, return_counts=True))


def get_count_attrib_correctness(data, index):
    """
    returns count success rate based on value for ith attribute in data
    :param data: data set
    :param index: count success rate based on value for ith attribute in data
    :return:
    """
    success = np.array([[0, 1], [0, 0]])
    for i in range(0, np.shape(data)[0]):
        if int(data[i][-1]) == 0:
            success[1][0] = success[1][0] + 1
        else:
            success[1][1] = success[1][1] + 1
    return success


def split_data_set(data, index, val):
    """
    keep only row which has given value for the column
    :param data: dataset on which operation need to be done
    :param index: index of the data on which operation need to be done
    :param val: value to retain for the data
    :return:
    """
    new_data = []
    for d in data:
        if int(d[index]) == int(val):
            new_data.append(d)
    return new_data


def variance_impurity(data):
    """
    calculate variance impurity for the data
    :param data: data for the the single attribute
    :return: variance impurity for the data
    """
    val, val_freq = np.unique(data[:, -1], return_counts=True)
    val_probability = val_freq / len(data)
    if len(val_probability) == 2:
        variance_imp = val_probability[0] * val_probability[1]
    else:
        variance_imp = 0.0
    return variance_imp


def entropy(data, entropy_or_var_imp):
    """
    Calculate entropy for an attribute
    :param data: data for the the single attribute
    :param entropy_or_var_imp: boolean attribute to denote which heuristic to use, True for Entropy and False for
            Variance Impurity
    :return: entropy in float representation
    """
    if entropy_or_var_imp:
        val, val_freq = np.unique(data[:, -1], return_counts=True)
        val_probability = val_freq / len(data)
        attr_entropy = -val_probability.dot(np.log2(val_probability))
        # print("entropy", attr_entropy)
        return attr_entropy
    else:
        return variance_impurity(data)


def heuristic_function(data, m, entropy_or_var_imp = True):
    """
    Calculate Information Gain for a attribute
    :param data: data set on which heuristic to be done
    :param m : number of features
    :param entropy_or_var_imp: boolean attribute to denote which heuristic to use, True for Entropy and False for
            Variance Impurity
    :return: information gain in float representation
    """
    base_entropy = entropy(data, entropy_or_var_imp)
    best_feature = None
    best_info_gain = -1.0
    for i in range(0, m):
        attribute_frequency = get_count_attrib(data[:, i])
        entr = 0.0
        for j in attribute_frequency[0]:
            new_data = np.array(split_data_set(data, i, j))
            prob = len(new_data) / float(len(data))
            entr += prob * entropy(new_data, entropy_or_var_imp)
        info_gain = base_entropy - entr
        if best_info_gain < info_gain:
            best_info_gain = info_gain
            best_feature = i

    # print(best_feature, best_info_gain, info_gain)
    return best_feature


def create_decision_tree(data, labels, heuristic= 'entropy'):
    """
    generate decision tree using the training data
    :param data: data set using which decision tree need to created
    :param labels: column label for the data set
    :param heuristic: heuristic to be used to generate Decision tree
    :return: decision tree
    """

    # no of features
    m = np.shape(data)[1] - 1
    target_val = [ex[-1] for ex in data]

    # return the first element if all are +ve or -ve case
    if target_val.count(target_val[0]) == len(target_val):
        new_node = Tree()
        new_node.data = target_val[0]
        if int(target_val[0]) == 0:
            new_node.accuracy = [str(len(target_val)) + "-"]
        else:
            new_node.accuracy = [str(len(target_val)) + "+"]
        return new_node

    if m == 0:
        return None

    if heuristic == 'entropy':
        best_feature = heuristic_function(data, m, True)
    else:
        best_feature = heuristic_function(data, m, False)
    best_feature_label = labels[best_feature]
    best_feature_frequency = get_count_attrib_correctness(data, best_feature)
    decision_tree_node = Tree()
    decision_tree_node.data = best_feature_label
    decision_tree_node.accuracy = np.array([str(best_feature_frequency[1][0])+"-", str(best_feature_frequency[1][1])+"+"])

    labels = np.delete(labels, best_feature)
    for val in best_feature_frequency[0]:
        refined_data = split_data_set(data, best_feature, val)
        if np.shape(refined_data)[0] != 0:
            refined_data = np.delete(refined_data, best_feature, axis=1)
            b = create_decision_tree(refined_data, labels, heuristic)
            if b is not None:
                decision_tree_node.child[int(val)] = b
    return decision_tree_node


def print_decision_tree_graph(generated_decision_tree_graph, depth=0):
    """
    print the decision tree
    :param generated_decision_tree_graph: decision tree which need to be printed
    :param depth: to keep track of depth (optional)
    """
    if generated_decision_tree_graph is None or generated_decision_tree_graph.child[0] is None:
        return
    else:
        # for index, child in enumerate(generated_decision_tree_graph.child):
        #     print("| " * depth + generated_decision_tree_graph.data, "=", index, ":",
        #         child.accuracy, ":", generated_decision_tree_graph.rank)
        #     print_decision_tree_graph(child, depth + 1)
        i = len(generated_decision_tree_graph.child)
        while i > 0:
            i = i - 1
            tar = ""
            if generated_decision_tree_graph.child[i] is not None and len(generated_decision_tree_graph.child[i].accuracy) == 1:
                if str(generated_decision_tree_graph.child[i].accuracy[0])[-1] == "-":
                    tar = 1
                else:
                    tar = 0
            print("| " * depth + generated_decision_tree_graph.data, "=", i, ":", tar)
            print_decision_tree_graph(generated_decision_tree_graph.child[i], depth + 1)


def load_data_set(filename):
    """
    load the data set in memory
    :param filename: path for the data set
    :return: returns the data set
    """
    data_set = np.array(pd.read_csv(filename, header=None))
    return data_set[1:, :], data_set[0, :-1]


def test_decision_tree(decision_tree, test_data, test_label):
    """
    test the decision tree
    :param decision_tree: generated decision tree
    :param test_data: test data set for testing accuracy
    """
    accuracy = 0
    i = 1
    # print(len(test_data))
    for data in test_data:
        tree = decision_tree
        while True:
            index = np.where(test_label == tree.data)[0]
            if len(index) == 1 and tree.child[int(data[index])] is not None:
                tree = tree.child[int(data[index])]
            else:
                break
        if tree.data == data[-1]:
           accuracy = accuracy + 1

    return round(accuracy * 100/len(test_data), 3)



def dfs(graph, start, visited=None):
    """

    :param graph:
    :param start:
    :param visited:
    :return:
    """
    global rank
    global node_ranking
    if start.data != str(0) and start.data != str(1):
        start.rank = rank
        node_ranking[rank] = start
        rank = rank + 1
    start.is_visited = True
    for index, child in enumerate(start.child):
        if child is not None and not child.is_visited:
            dfs(graph, child, visited)
    return graph


def prune(decision_tree_copy_prune, validation_set, validation_labels, m):
    """

    :param decision_tree_copy_prune:
    :param validation_set:
    :param validation_labels:
    :param m:
    :return:
    """
    for j in range(1, m):
        n = rank - 1
        p = random.randint(1, n)
        visited, stack = set(), [decision_tree_copy_prune]
        while stack and p > 0:
            p = p - 1
            start_node = stack.pop()
            if start_node not in visited:
                visited.add(start_node)
                for child in start_node.child:
                    if child is not None and child not in visited:
                        stack.append(child)

        if len(start_node.accuracy) == 2 and start_node.data != 1 and start_node.data != 0:
            neg = int(start_node.accuracy[0][:-1])
            pos = int(start_node.accuracy[1][:-1])
            if neg > pos:
                start_node.data = 0
            else:
                start_node.data = 1
            start_node.child[0] = None
            start_node.child[1] = None
    accuracy = test_decision_tree(decision_tree_copy_prune, validation_set, validation_labels)
    return accuracy


def post_pruning(decision_tree_copy, l, k, validation_set, validation_labels, old_accuracy):
    """

    :param decision_tree_copy:
    :param l:
    :param k:
    :param validation_set:
    :param validation_labels:
    :param old_accuracy:
    :return:
    """
    best_decision_tree = None
    global rank
    for i in range(1, int(l)):
        temp_decision_tree = decision_tree_copy
        m = random.randint(1, int(k))
        updated_accuracy = prune(temp_decision_tree, validation_set, validation_labels, m)
        if old_accuracy < updated_accuracy:
            best_decision_tree = temp_decision_tree
            old_accuracy = updated_accuracy
    return best_decision_tree


# ------------------------ Main Block ----------------------- #
if sys.argv.__len__() == 7:

    L = sys.argv[1]
    K = sys.argv[2]
    training_set_filename = sys.argv[3]
    validation_set_filename = sys.argv[4]
    test_set_filename = sys.argv[5]
    to_print = sys.argv[6]

    # loading data sets
    training_set, labels = load_data_set(training_set_filename)
    test_set, test_labels = load_data_set(test_set_filename)
    validation_set, validation_labels = load_data_set(validation_set_filename)

    print('---------------------- Information Gain Heuristic -------------------')
    generated_decision_tree_graph = create_decision_tree(training_set, labels, 'entropy')
    generated_decision_tree_graph = dfs(generated_decision_tree_graph, generated_decision_tree_graph)
    if to_print.lower() == 'yes':
        print_decision_tree_graph(generated_decision_tree_graph)

    acc = test_decision_tree(generated_decision_tree_graph, validation_set, validation_labels)
    print('Efficiency for Decision tree before pruning')
    print('Accuracy for the Validation Data', acc)
    print('Accuracy for the Test Data', test_decision_tree(generated_decision_tree_graph, test_set, test_labels))

    updated_decision_tree = post_pruning(generated_decision_tree_graph, L, K, validation_set, validation_labels, acc)
    if updated_decision_tree is None:
        updated_decision_tree = generated_decision_tree_graph
    new_acc = test_decision_tree(updated_decision_tree, validation_set, test_labels)
    print('Accuracy for the decision tree post pruning is', new_acc)

    print()
    print('---------------------- Variance Impurity Heuristic -------------------')
    generated_decision_tree_graph_vi = create_decision_tree(training_set, labels, 'var_imp')
    generated_decision_tree_graph_vi = dfs(generated_decision_tree_graph_vi, generated_decision_tree_graph_vi)
    if to_print.lower() == 'yes':
        print_decision_tree_graph(generated_decision_tree_graph_vi)

    acc = test_decision_tree(generated_decision_tree_graph_vi, validation_set, validation_labels)
    print('Efficiency for Decision tree before pruning')
    print('Accuracy for the Validation Data', acc)
    print('Accuracy for the Test Data', test_decision_tree(generated_decision_tree_graph_vi, test_set, test_labels))
    updated_decision_tree_vi = post_pruning(generated_decision_tree_graph_vi, L, K, validation_set, validation_labels, acc)
    if updated_decision_tree_vi is None:
        updated_decision_tree_vi = generated_decision_tree_graph_vi
    new_acc = test_decision_tree(updated_decision_tree_vi, validation_set, test_labels)
    print('Accuracy for the decision tree post pruning is', new_acc)

else:
    print("Invalid number of arguments. Arguments required: <L> <K>",
          "<training-set> <validation-set> <test-set> <to-print>")
