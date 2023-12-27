"""Machine learning algorithm implementations.

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from decision_tree import *

from random import randint
import math


#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------


def random_subset(F, columns):
    """Returns F unique column names from the given list of columns. The
    column names are selected randomly from the given names.

    Args: 
        F: The number of columns to return.
        columns: The columns to select F column names from.

    Notes: If F is greater or equal to the number of names in columns,
       then the columns list is just returned.

    """
    col_copy = columns.copy()
    rand_subset = []

    # Check if F is greater or equal to number of names
    if F >= len(columns):
        return columns

    # Randomly select F number of columns using rand_idx
    for i in range(F):
        rand_idx = randint(0, len(col_copy) - 1)
        rand_subset.append(col_copy[rand_idx])
        del col_copy[rand_idx]
    
    return rand_subset


def tdidt_F(table, label_col, F, columns): 
    """Returns an initial decision tree for the table using information
    gain, selecting a random subset of size F of the columns for
    attribute selection. If fewer than F columns remain, all columns
    are used in attribute selection.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        F: The number of columns to randomly subselect
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # Fewer than F columns remain case
    if len(columns) < F:
        return tdidt(table, label_col, columns)
    
    # Random subset of F size decision tree case
    rand_set = random_subset(F, columns)
    return tdidt(table, label_col, rand_set)



def closest_centroid(centroids, row, columns):
    """Given k centroids and a row, finds the centroid that the row is
    closest to.

    Args:
        centroids: The list of rows serving as cluster centroids.
        row: The row to find closest centroid to.
        columns: The numerical columns to calculate distance from. 
    
    Returns: The index of the centroid the row is closest to. 

    Notes: Uses Euclidean distance (without the sqrt) and assumes
        there is at least one centroid.

    """
    # Initialize the closest centroid and corresponding distance
    closest_idx = None
    closest_sq_dist = float('inf')

    # Traverse the centroids and compare row distances
    for i in range(len(centroids)):

        # Check if row is one of the centroids
        if centroids[i] == row:
            return i
        
        # Compute the square distances with current centroid
        curr_dist = 0
        for col in columns:
            curr_dist += (centroids[i][col] - row[col])**2

        # If current distance is less than closest distance, set new
        if curr_dist < closest_sq_dist:
            closest_idx = i
            closest_sq_dist = curr_dist

    return closest_idx



def select_k_random_centroids(table, k):
    """Returns a list of k random rows from the table to serve as initial
    centroids.

    Args: 
        table: The table to select rows from.
        k: The number of rows to select values from.
    
    Returns: k unique rows. 

    Notes: k must be less than or equal to the number of rows in the table. 

    """
    # Initialize list of centroids and copy of table to drop used rows from
    centroid_list = []
    centroid_table = table.copy()

    # Iterate k times the selection of random index and addition to centroid list
    for i in range(k):
        rand_idx = randint(0, centroid_table.row_count() - 1)
        centroid_list.append(centroid_table[rand_idx])
        del centroid_table[rand_idx]

    return centroid_list



def k_means(table, centroids, columns): 
    """Returns k clusters from the table using the initial centroids for
    the given numerical columns.

    Args:
        table: The data table to build the clusters from.
        centroids: Initial centroids to use, where k is length of centroids.
        columns: The numerical columns for calculating distances.

    Returns: A list of k clusters, where each cluster is represented
        as a data table.

    Notes: Assumes length of given centroids is number of clusters k to find.

    """
    # Initialize a cluster list of empty data tables and a bool to check if centroids have moved
    cluster_list = [DataTable(table.columns()) for i in range(len(centroids))]
    centroids_move = True
    
    # Repeat row assignment until centroids no longer move
    while(centroids_move == True):

        # Initialize list of data tables for each centroid
        curr_list = [DataTable(table.columns()) for i in range(len(centroids))]

        # Traverse the table and add rows to respective cluster table
        for row in table:
            # Calculate the index of closest centroid
            c_idx = closest_centroid(centroids, row, columns)

            # Append row to cluster table
            curr_list[c_idx].append(row.values())
        
        # Check if clusters have changed
        for i in range(len(cluster_list)):
            # Check if clusters have the same points in them
            if cluster_list[i].row_count() != curr_list[i].row_count():
                centroids_move = True
                break
            else:
                centroids_move = False
        
        # Update cluster list and centroids
        cluster_list = curr_list
        for i in range(len(cluster_list)):
            centroids[i] = DataRow(columns, [mean(cluster_list[i], col) for col in columns])
            
    return cluster_list



def tss(clusters, columns):
    """Return the total sum of squares (tss) for each cluster using the
    given numerical columns.

    Args:
        clusters: A list of data tables serving as the clusters
        columns: The list of numerical columns for determining distances.
    
    Returns: A list of tss scores for each cluster. 

    """
    # Create empty list for tss for each cluster
    tss_list = []

    # Traverse clusters and calculate tss
    for cluster in clusters:

        # Calculate cluster's centroid and initialize tss variable
        curr_centroid = DataRow(columns, [mean(cluster, col) for col in columns])
        cluster_tss = 0

        # Calculate square sum of each row
        for row in cluster:
            curr_sq_sum = 0
            for col in columns:
                curr_sq_sum += (row[col] - curr_centroid[col])**2
            cluster_tss += curr_sq_sum
        
        # Append current tss to list
        tss_list.append(cluster_tss)
    
    return tss_list



#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------

def same_class(table, label_col):
    """Returns true if all of the instances in the table have the same
    labels and false otherwise.

    Args: 
        table: The table with instances to check. 
        label_col: The column with class labels.

    """
    # Find the label of first instance in table
    label_check = table[0][label_col]

    # Traverse table and check if each label is equivalent to first
    for row in table:
        if row[label_col] != label_check:
            return False
    
    return True


def build_leaves(table, label_col):
    """Builds a list of leaves out of the current table instances.
    
    Args: 
        table: The table to build the leaves out of.
        label_col: The column to use as class labels

    Returns: A list of LeafNode objects, one per label value in the
        table.

    """
    # Check if table is empty and return empty list
    if table.row_count() == 0:
        return []
    
    # Partition the table by label and create leaves based on partition
    leaf_node_list = []
    label_list = partition(table, [label_col])

    for part in label_list:
        leaf_node_list.append(LeafNode(part[0][label_col], part.row_count(), table.row_count()))

    return leaf_node_list


def calc_e_new(table, label_col, columns):
    """Returns entropy values for the given table, label column, and
    feature columns (assumed to be categorical). 

    Args:
        table: The table to compute entropy from
        label_col: The label column.
        columns: The categorical columns to use to compute entropy from.

    Returns: A dictionary, e.g., {e1:['a1', 'a2', ...], ...}, where
        each key is an entropy value and each corresponding key-value
        is a list of attributes having the corresponding entropy value. 

    Notes: This function assumes all columns are categorical.

    """
    # Prepare variables to calculate |Di|/|D| x e_j for each partition
    total_instances = table.row_count()
    e_new_dict = dict()

    # Traverse each feature and calculate e_new for given feature
    for feature in columns:

        curr_e_new = 0

        # Partition table by current feature
        feature_part = partition(table, [feature])

        # Calculate e_j by partitioning label partition
        for value_table in feature_part:

            # Find the total number of instances with certain value
            total_value_instances = value_table.row_count()
            e_j = 0

            # Find the e_j subsection by label partition of current feature
            value_label_part = partition(value_table, [label_col])

            for val_label in value_label_part:
                val_label_size = val_label.row_count() / total_value_instances
                e_j += -(val_label_size * math.log2(val_label_size))
            
            curr_e_new += ((value_table.row_count() / total_instances) * e_j)
        
        # Add newly calculated e_new to the dictionary as new key or append current feature to existing key
        if curr_e_new not in e_new_dict.keys():
            e_new_dict[curr_e_new] = [feature]
        else:
            e_new_dict[curr_e_new] += [feature]

    return e_new_dict



def tdidt(table, label_col, columns): 
    """Returns an initial decision tree for the table using information
    gain.

    Args:
        table: The table to build the decision tree from. 
        label_col: The column containing class labels. 
        columns: The categorical columns. 

    Notes: The resulting tree can contain multiple leaf nodes for a
        particular attribute value, may have attributes without all
        possible attribute values, and in general is not pruned.

    """
    # Base Case 1: Check if table is empty
    if table.row_count() == 0:
        return None

    # Base Case 2: Check if table instances are all of the same class, return single leaf node
    if same_class(table, label_col):
        return [LeafNode(table[0][label_col], table.row_count(), table.row_count())]

    # Base Case 3: If no more attributes to partition on, return leaves from current partition
    if len(columns) == 0:
        return build_leaves(table, label_col)

    # Building a new Attribute Node (RECURSIVE SECTION)
    # Find E_new values for each column
    e_new_dict = calc_e_new(table, label_col, columns)

    # Find column with smallest E_new value
    min_e_new = min(e_new_dict.keys())
    min_e_new_column = e_new_dict[min_e_new][0]

    # Partition on the column
    column_partition = partition(table, [min_e_new_column])

    # Create attribute node and fill in value nodes (recursive calls on partition)
    attribute_dict = dict()
    new_columns = []
    for col in columns:
        if col != min_e_new_column:
            new_columns.append(col)
    for col_val in column_partition:
        attribute_dict[col_val[0][min_e_new_column]] = tdidt(col_val, label_col, new_columns)

    # Return the newly created attribute node
    return AttributeNode(min_e_new_column, attribute_dict)


def summarize_instances(dt_root):
    """Return a summary by class label of the leaf nodes within the given
    decision tree.

    Args: 
        dt_root: The subtree root whose (descendant) leaf nodes are summarized. 

    Returns: A dictionary {label1: count, label2: count, ...} of class
        labels and their corresponding number of instances under the
        given subtree root.

    """
    # Base Case: The root is a leaf node
    if type(dt_root) == LeafNode:
        return {dt_root.label: dt_root.count}
    
    # Base Case: The root is a list of leaf nodes
    if type(dt_root) == list:

        # Fill new dictionary with each leaf node count
        leaf_dict = {}
        for leaf in dt_root:
            leaf_dict[leaf.label] = leaf.count
        
        return leaf_dict
    
    # Recursive Step: The root is an attribute node and each node must be summarized
    root_dict = dt_root.values
    attribute_dict = {}

    # Traverse the attribute node's values
    for node in root_dict.values():

        # Grab the summarized dictionary of current node
        temp_dict = summarize_instances(node)

        # Append counts to master dictionary, checking if key is already present
        for key, value in temp_dict.items():
            if key not in list(attribute_dict.keys()):
                attribute_dict[key] = value
            else:
                attribute_dict[key] += value
        
    return attribute_dict


def resolve_leaf_nodes(dt_root):
    """Modifies the given decision tree by combining attribute values with
    multiple leaf nodes into attribute values with a single leaf node
    (selecting the label with the highest overall count).

    Args:
        dt_root: The root of the decision tree to modify.

    Notes: If an attribute value contains two or more leaf nodes with
        the same count, the first leaf node is used.

    """
    # Base Case 1: Root is a leaf node, return copy of dt_root
    if type(dt_root) == LeafNode:
        return dt_root

    # Base Case 2: Root is a list of leaf nodes
    if type(dt_root) == list: 
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]

    # Recursive Step
    # Create a new decision tree attribute node (same name as dt_root)
    new_dt_root = AttributeNode(dt_root.name, {})

    # Navigate the tree recursively
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_leaf_nodes(child)

    # Backtracking Phase: look at each attribute child
    # For each new_dt_root value, combine its leaves if it has multiple
    for val, child in new_dt_root.values.items():

        # Check for multiple leaves and perform combine
        if type(child) == list:

            # Find the highest leaf count and set it to current value in dictionary
            max_leaf = None
            max_count = 0
            for leaf in child:
                if leaf.count > max_count:
                    max_leaf = leaf
                    max_count = leaf.count
            
            new_dt_root.values[val] = [LeafNode(max_leaf.label, max_leaf.count, max_leaf.total)]


    # Return the new_dt_root
    return new_dt_root


def resolve_attribute_values(dt_root, table):
    """Return a modified decision tree by replacing attribute nodes
    having missing attribute values with the corresponding summarized
    descendent leaf nodes.
    
    Args:
        dt_root: The root of the decision tree to modify.
        table: The data table the tree was built from. 

    Notes: The table is only used to obtain the set of unique values
        for attributes represented in the decision tree.

    """
    # Base Case 1: Root is a leaf node, return copy of dt_root
    if type(dt_root) == LeafNode:
        return dt_root

    # Base Case 2: Root is a list of leaf nodes
    if type(dt_root) == list: 
        return [LeafNode(l.label, l.count, l.total) for l in dt_root]

    # Recursive Step
    # Create a new decision tree attribute node (same name as dt_root)
    new_dt_root = AttributeNode(dt_root.name, {})

    # Navigate the tree recursively
    for val, child in dt_root.values.items():
        new_dt_root.values[val] = resolve_attribute_values(child, table)

    # Backtracking Phase: look at each attribute child
    # For each new_dt_root value, return list of leaf nodes if there is missing attribute value
    unique_attr_values = distinct_values(table, [dt_root.name])
    for val in new_dt_root.values.keys():
        if val in unique_attr_values:
            unique_attr_values.remove(val)

    # Determine if there are any missing values within current root
    if len(unique_attr_values) > 0:
        
        # If true, create a list of leaf nodes that provide a summary of the current node instances
        leaf_list = []
        label_dict = summarize_instances(dt_root)
        for label, count in label_dict.items():
            leaf_list.append(LeafNode(label, count, sum(label_dict.values())))

        return leaf_list

    # Return the new_dt_root
    return new_dt_root


def tdidt_predict(dt_root, instance): 
    """Returns the class for the given instance given the decision tree. 

    Args:
        dt_root: The root node of the decision tree. 
        instance: The instance to classify. 

    Returns: A pair consisting of the predicted label and the
       corresponding percent value of the leaf node.

    """
    # Base Case: if root is a leaf node
    if type(dt_root) == LeafNode:
        return (dt_root.label, dt_root.percent())
    
    # Base Case: if root is a list of one singular leaf node
    if type(dt_root) == list:
        return (dt_root[0].label, dt_root[0].percent())
    
    # Recursion: traverse down the tree using the name of next attribute to move
    inst_col_node = instance[dt_root.name] 
    return tdidt_predict(dt_root.values[inst_col_node], instance)

#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def naive_bayes(table, instance, label_col, continuous_cols, categorical_cols=[]):
    """Returns the labels with the highest probabibility for the instance
    given table of instances using the naive bayes algorithm.

    Args:
       table: A data table of instances to use for estimating most probably labels.
       instance: The instance to classify.
       continuous_cols: The continuous columns to use in the estimation.
       categorical_cols: The categorical columns to use in the estimation. 

    Returns: A pair (labels, prob) consisting of a list of the labels
        with the highest probability and the corresponding highest
        probability.

    """
    # Create a dictionary of class labels to store their corresponding probabilities
    class_dict = dict()

    # Partition train set by class label
    label_partition = partition(table, [label_col])

    # Traverse partition and look at each class label to calculate its probability
    for class_part in label_partition:

        # Calculate the percent of current class label instances in train set: P(C)
        c_prob = class_part.row_count() / table.row_count()

        # Find the independent probabilities of each attribute in partition
        v_attr_prob_list = []

        # Add categorical values to probability list using independence assumption
        for attribute in categorical_cols:

            match_count = 0
            for row in class_part:
                if instance[attribute] == row[attribute]:
                    match_count += 1
            
            v_attr_prob_list.append(match_count / class_part.row_count())

        # Use the Gaussian formula to calculate the probability of independent continuous attribute
        for attribute in continuous_cols:

            # Calculate Gaussian Density and append to probability list
            attr_mean = mean(class_part, attribute)
            attr_std = std_dev(class_part, attribute)
            gaus_prob = gaussian_density(instance[attribute], attr_mean, attr_std)

            v_attr_prob_list.append(gaus_prob)
        
        # Multiply all the individual probabilities together to calculate P(X | C)
        prob = 1
        for i in v_attr_prob_list:
            prob *= i
        
        prob = prob * c_prob

        # Check if probability has any class labels currently: append or create new list
        if prob not in class_dict.keys():
            class_dict[prob] = [class_part[0][label_col]]
        else:
            class_dict[prob] += [class_part[0][label_col]]

    # Find labels with the highest probability and return their probability pair
    high_prob = max(class_dict)
    return (class_dict[high_prob], high_prob)


def gaussian_density(x, mean, sdev):
    """Return the probability of an x value given the mean and standard
    deviation assuming a normal distribution.

    Args:
        x: The value to estimate the probability of.
        mean: The mean of the distribution.
        sdev: The standard deviation of the distribution.

    """
    return (1 / (sqrt(2 * math.pi) * sdev)) * (math.e ** ((-((x - mean) ** 2)) / (2 * (sdev ** 2))))


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def knn(table, instance, k, numerical_columns, nominal_columns=[]):
    """Returns the k closest distance values and corresponding table
    instances forming the nearest neighbors of the given instance. 

    Args:
        table: The data table whose instances form the nearest neighbors.
        instance: The instance to find neighbors for in the table.
        k: The number of closest distances to return.
        numerical_columns: The numerical columns to use for comparison.
        nominal_columns: The nominal columns to use for comparison (if any).

    Returns: A dictionary with k key-value pairs, where the keys are
        the distances and the values are the corresponding rows.

    Notes: 
        The numerical and nominal columns must be disjoint. 
        The numerical and nominal columns must be valid for the table.
        The resulting score is a combined distance without the final
        square root applied.

    """
    # Create empty dictionary to hold k key-value pairs
    row_dist = dict()

    # Traverse table to calculate each row distance 
    for row in table:
        distance = 0

        # Calculate numerical column square differences
        for num_label in numerical_columns:
            cur_dif_sq = (row[num_label] - instance[num_label]) ** 2
            distance = distance + cur_dif_sq

        # Calculate nominal column square diffferences
        for nom_label in nominal_columns:
            if instance[nom_label] != row[nom_label]:
                distance = distance + 1
        
        # Check if distance has any row values currently: append or create new list
        if distance not in row_dist.keys():
            row_dist[distance] = [row]
        else:
            row_dist[distance] += [row]

    # Sort dictionary items and create return dictionary for k number of values
    sorted_dist_items = sorted(row_dist.items())
    k_row_dist = dict()

    # Add first k elements of sorted dictionary to return dictionary
    for key, value in sorted_dist_items:
        k_row_dist[key] = value
        if len(k_row_dist) == k:
            break

    return k_row_dist

def majority_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances that occur the most.

    Args:
        instances: A list of instance rows.
        labeled_column: The column holding the class labels.

    Returns: A list of the labels that occur the most in the given
    instances.

    """
    # Create a dictionary of vote counts for each label
    label_counts = dict()

    # Add each vote to dictionary where label is key and value is vote count
    for row in instances:

        vote = row[labeled_column]

        if vote not in label_counts.keys():
            label_counts[vote] = 1
        else:
            label_counts[vote] = label_counts[vote] + 1

    # Traverse dictionary and append keys with max value to list
    majority_list = []
    for key, value in label_counts.items():
        if value == max(label_counts.values()):
            majority_list.append(key)

    return majority_list



def weighted_vote(instances, scores, labeled_column):
    """Returns the labels in the given instances with the largest total
    sum of corresponding scores.

    Args:
        instances: The list of instance rows.
        scores: The corresponding scores for each instance.
        labeled_column: The column with class labels.

    """
    # Create a dictionary of aggregate scores for each label
    label_scores = dict()

    # Add each vote score to dictionary where label is key and value is aggregate score
    for i in range(len(instances)):

        vote = instances[i][labeled_column]

        if vote not in label_scores.keys():
            label_scores[vote] = scores[i]
        else:
            label_scores[vote] = label_scores[vote] + scores[i]

    # Traverse dictionary and append keys with max value to list
    majority_list = []
    for key, value in label_scores.items():
        if value == max(label_scores.values()):
            majority_list.append(key)

    return majority_list