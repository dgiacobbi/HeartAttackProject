"""Machine learning algorithm evaluation functions. 

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

from data_table import *
from data_util import *
from data_learn import *
from random import randint



#----------------------------------------------------------------------
# HW-8
#----------------------------------------------------------------------

def bootstrap(table): 
    """Creates a training and testing set using the bootstrap method.

    Args: 
        table: The table to create the train and test sets from.

    Returns: The pair (training_set, testing_set)

    """
    # Create DataTable objects for train and test sets
    train = DataTable(table.columns())
    test = DataTable(table.columns())

    # Randomly select |D| instances from dataset D to build train set
    used_idx = []
    for i in range(table.row_count()):
        # Put random instance into train set
        rand_idx = randint(0, table.row_count()-1)
        train.append(table[rand_idx].values())
        used_idx.append(rand_idx)
    
    # Add remaining rows not added to train into test set
    for j in range(table.row_count()):
        if j not in used_idx:
            test.append(table[j].values())

    return train, test



def stratified_holdout(table, label_col, test_set_size):
    """Partitions the table into a training and test set using the holdout
    method such that the test set has a similar distribution as the
    table.

    Args:
        table: The table to partition.
        label_col: The column with the class labels. 
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    # Create the train and test sets to return
    train = DataTable(table.columns())
    test = DataTable(table.columns())

    # Parition the table by label
    label_partition = partition(table, [label_col])

    # Calculate the number of instances for each partition in test function
    part_counts = []
    for label in label_partition:
        part_distribution = label.row_count() / table.row_count()
        part_counts.append(int(part_distribution*test_set_size))

    # Add elements of each partition to test set until full
    for l in range(len(label_partition)):
        # Find the partition's corresponding count and randomly add instances to test
        for i in range(part_counts[l]):
            idx_add = randint(0, label_partition[l].row_count()-1)
            test.append(label_partition[l][idx_add].values())
            del label_partition[l][idx_add]


    # Add the remaining rows into the training set
    for label_part in label_partition:
        for row in label_part:
            train.append(row.values())

    return (train, test)
    


def tdidt_eval_with_tree(dt_root, test, label_col, table):
    """Evaluates the given test set using tdidt over the dt_root,
       returning a corresponding confusion matrix.

    Args:
       dt_root: The decision tree to use.
       test: The testing data set.
       label_col: The column being predicted.
       table: used to grab label column values for both test and tree

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    """
    # Build matrix with column labels
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform decision tree on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform tdidt on current test instance
        predict_label, pred_prob = tdidt_predict(dt_root, test_row)

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix



def random_forest(table, remainder, F, M, N, label_col, columns):
    """Returns a random forest build from the given table. 
    
    Args:
        table: The original table for cleaning up the decision tree.
        remainder: The table to use to build the random forest.
        F: The subset of columns to use for each classifier.
        M: The number of unique accuracy values to return.
        N: The total number of decision trees to build initially.
        label_col: The column with the class labels.
        columns: The categorical columns used for building the forest.

    Returns: A list of (at most) M pairs (tree, accuracy) consisting
        of the "best" decision trees and their corresponding accuracy
        values. The exact number of trees (pairs) returned depends on
        the other parameters (F, M, N, and so on).

    """
    # Create bootstrap samples using the remainder set
    n_trees = []
    for i in range(N):
        # Create a bootstrap sample from the remainder
        training, validation = bootstrap(remainder)

        # Build decision tree with training set and prune as necessary
        curr_tree = tdidt_F(training, label_col, F, columns)
        curr_tree = resolve_attribute_values(curr_tree, table)
        curr_tree = resolve_leaf_nodes(curr_tree)

        if validation.row_count() == 0:
            n_trees.append((curr_tree, 0))
            continue

        # Evaluate decision tree and calculate accuracy
        curr_matrix = tdidt_eval_with_tree(curr_tree, validation, label_col, table)

        # Compute accuracy, precision, recall
        acc = []
        for label in distinct_values(table, [label_col]):
            acc.append(accuracy(curr_matrix, label))
        curr_acc = sum(acc)/len(acc)

        # Append (tree, accuracy) to n_trees list
        n_trees.append((curr_tree, curr_acc))
    
    # Pick M best classifiers
    m_trees = []
    for j in range(M):
        # Check if n_trees list is empty
        if len(n_trees) == 0:
            break
        # Determine the tree with current highest accuracy
        max_tree = None
        max_acc = 0
        for tree in n_trees:
            if tree[1] > max_acc:
                max_tree = tree
                max_acc = tree[1]
        # Remove highest tree from n_trees and add to m_trees
        n_trees.remove(max_tree)
        m_trees.append(max_tree)

    return m_trees



def random_forest_eval(table, remainder, test, F, M, N, label_col, columns):
    """Builds a random forest and evaluate's it given a training and
    testing set.

    Args: 
        table: The initial table.
        remainder: The training set from the initial table.
        test: The testing set from the initial table.
        F: Number of features (columns) to select.
        M: Number of trees to include in random forest.
        N: Number of trees to initially generate.
        label_col: The column with class labels. 
        columns: The categorical columns to use for classification.

    Returns: A confusion matrix containing the results. 

    Notes: Assumes weighted voting (based on each tree's accuracy) is
        used to select predicted label for each test row.

    """
    # Create N bootstrap samples from remainder and get M best classifiers
    m_trees = random_forest(table, remainder, F, M, N, label_col, columns)

     # Build matrix with column labels
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list
    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform decision tree on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Determine the label using weighted voting on the m_trees to create dictionary
        m_predictions = {}
        for tree, acc in m_trees:
            predict_label, predict_percent = tdidt_predict(tree, test_row)
            
            if predict_label not in list(m_predictions.keys()):
                m_predictions[predict_label] = acc
            else:
                m_predictions[predict_label] += acc
        
        # Find the label with the highest voting score
        max_label = None
        max_acc = 0
        for key, value in m_predictions.items():
            if value > max_acc:
                max_label = key
                max_acc = value

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][max_label] = conf_matrix[cur_row_index][max_label] + 1

    return conf_matrix



#----------------------------------------------------------------------
# HW-7
#----------------------------------------------------------------------
def tdidt_eval(train, test, label_col, columns):
    """Evaluates the given test set using tdidt over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       columns: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)
    
    # Build decision tree
    predict_tree = tdidt(train, label_col, columns)
    clean_tree = resolve_attribute_values(predict_tree, train)
    clean_tree = resolve_leaf_nodes(clean_tree)

    # Perform decision tree on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform tdidt on current test instance
        predict_label, pred_prob = tdidt_predict(clean_tree, test_row)

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def tdidt_stratified(table, k_folds, label_col, columns):
    """Evaluates tdidt prediction approach over the table using stratified
    k-fold cross validation, returning a single confusion matrix of
    the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        columns: The categorical columns for tdidt. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Build decision tree with train set
        predict_tree = tdidt(train, label_col, columns)
        attr_predict_tree = resolve_attribute_values(predict_tree, train)
        final_predict_tree = resolve_leaf_nodes(attr_predict_tree)

        # Perform decision tree on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform decision tree on current test instance
            predict_label, pred_prob = tdidt_predict(final_predict_tree, test_row)

            # Update confusion matrix row with prediction
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


#----------------------------------------------------------------------
# HW-6
#----------------------------------------------------------------------

def stratify(table, label_column, k):
    """Returns a list of k stratified folds as data tables from the given
    data table based on the label column.

    Args:
        table: The data table to partition.
        label_column: The column to use for the label. 
        k: The number of folds to return. 

    Note: Does not randomly select instances for the folds, and
        instead produces folds in order of the instances given in the
        table.

    """
    # Partition the original table by class label
    label_partition = partition(table, [label_column])

    # Create a k-size list of empty data tables to distribute values
    stratify_list = [DataTable(table.columns()) for i in range(k)]

    # Traverse each class label partition
    for part in label_partition:

        # Bin Index to loop through stratify list
        bin_index = 0

        # Distribute rows with class label even across bins
        for row in part:

            stratify_list[bin_index].append(row.values())

            # Increment the bin index each time, looping back to 0 as needed
            bin_index += 1
            if bin_index == k:
                bin_index = 0

    return stratify_list


def union_all(tables):
    """Returns a table containing all instances in the given list of data
    tables.

    Args:
        tables: A list of data tables. 

    Notes: Returns a new data table that contains all the instances of
       the first table in tables, followed by all the instances in the
       second table in tables, and so on. The tables must have the
       exact same columns, and the list of tables must be non-empty.

    """
    # Check to make sure tables has at least one list
    if len(tables) == 0:
        raise ValueError

    # Create an empty data table to add values to
    union_table = DataTable(tables[0].columns())

    # Traverse tables list and add rows from each table
    for table in tables:

        # Check to make sure tables can be combined together
        if len(table.columns()) != len(union_table.columns()):
            raise ValueError

        for i in range(len(table.columns())):
            if table.columns()[i] != union_table.columns()[i]:
                raise ValueError

        # Add rows from current table to the union table
        for row in table:
            union_table.append(row.values())

    return union_table

def naive_bayes_eval(train, test, label_col, continuous_cols, categorical_cols=[]):
    """Evaluates the given test set using naive bayes over the training
    set, returning a corresponding confusion matrix.

    Args:
       train: The training data set.
       test: The testing data set.
       label_col: The column being predicted.
       continuous_cols: The continuous columns (estimated via PDF)
       categorical_cols: The categorical columns

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the naive bayes returns multiple labels for a given test
        instance, the first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform naive bayes on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform naive bayes on current test instance
        pred_labels, pred_prob = naive_bayes(train, test_row, label_col, continuous_cols, categorical_cols)

        # If more than one predict label in highest probability, select first
        predict_label = pred_labels[0]

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def naive_bayes_stratified(table, k_folds, label_col, cont_cols, cat_cols=[]):
    """Evaluates naive bayes over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        cont_cols: The continuous columns for naive bayes. 
        cat_cols: The categorical columns for naive bayes. 

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Perform naive bayes on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform naive bayes on current test instance
            pred_labels, pred_prob = naive_bayes(train, test_row, label_col, cont_cols, cat_cols)

            # If more than one predict label in highest probability, select first
            predict_label = pred_labels[0]

            # Update confusion matrix row with prediction
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


def knn_stratified(table, k_folds, label_col, vote_fun, k, num_cols, nom_cols=[]):
    """Evaluates knn over the table using stratified k-fold cross
    validation, returning a single confusion matrix of the results.

    Args:
        table: The data table.
        k_folds: The number of stratified folds to use.
        label_col: The column with labels to predict. 
        vote_fun: The voting function to use with knn.
        num_cols: The numeric columns for knn.
        nom_cols: The nominal columns for knn.

    Notes: Each fold created is used as the test set whose results are
        added to a combined confusion matrix from evaluating each
        fold.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(table, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Stratify the table into k-folds
    stratify_table = stratify(table, label_col, k_folds)

    # Traverse each table section in k-folds and perform naive bayes on it as the test set
    for i in range(len(stratify_table)):

        # Combine the other tables to make the train set
        train_tables = []
        for j in range(len(stratify_table)):
            if j != i:
                train_tables.append(stratify_table[j])
        train = union_all(train_tables)

        # Perform knn on the table using train and test set
        for test_row in stratify_table[i]:

            # Get current label's index in confusion matrix
            cur_row_index = label_list.index(test_row[label_col])

            # Perform knn on current test instance
            knn_dict = knn(train, test_row, k, num_cols, nom_cols)

            # Create a list of all knn instances and corresponding scores
            knn_instances = []
            knn_scores = []

            for key, value in knn_dict.items():
                for instance in value:
                    knn_instances.append(instance)         
                    knn_scores.append(key)
            
            # Get the predicition from voting function
            maj_label_list = vote_fun(knn_instances, knn_scores, label_col)
            predict_label = maj_label_list[0]

            # Update confusion matrix row with prediction
            conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix


#----------------------------------------------------------------------
# HW-5
#----------------------------------------------------------------------

def holdout(table, test_set_size):
    """Partitions the table into a training and test set using the holdout method. 

    Args:
        table: The table to partition.
        test_set_size: The number of rows to include in the test set.

    Returns: The pair (training_set, test_set)

    """
    # Create an empty test set and copy the table to train set
    train_set = table.copy()
    test_set = DataTable(table.columns())

    # Given test set size, move rows from train set to test set
    for i in range(test_set_size):

        # Pick a random row from train set
        rand_index = randint(0, train_set.row_count() - 1)

        test_set.append(train_set[rand_index].values())
        del train_set[rand_index]

    return(train_set, test_set)


def knn_eval(train, test, vote_fun, k, label_col, numeric_cols, nominal_cols=[]):
    """Returns a confusion matrix resulting from running knn over the
    given test set. 

    Args:
        train: The training set.
        test: The test set.
        vote_fun: The function to use for selecting labels.
        k: The k nearest neighbors for knn.
        label_col: The column to use for predictions. 
        numeric_cols: The columns compared using Euclidean distance.
        nominal_cols: The nominal columns for knn (match or no match).

    Returns: A data table with n rows (one per label), n+1 columns (an
        'actual' column plus n label columns), and corresponding
        prediction vs actual label counts.

    Notes: If the given voting function returns multiple labels, the
        first such label is selected.

    """
    # Initialize a confusion matrix of zeros
    label_list = distinct_values(train, [label_col])
    matrix_col = ['actual'] + label_list

    conf_matrix = DataTable(matrix_col)

    # Fill confusion matrix with zeros
    for label in label_list:
        temp_list = [label] + [0 for j in range(len(label_list))]
        conf_matrix.append(temp_list)

    # Perform knn on the table using train and test set
    for test_row in test:

        # Get current label's index in confusion matrix
        cur_row_index = label_list.index(test_row[label_col])

        # Perform knn on current test instance
        knn_dict = knn(train, test_row, k, numeric_cols, nominal_cols)

        # Create a list of all knn instances and corresponding scores
        knn_instances = []
        knn_scores = []

        for key, value in knn_dict.items():
            for instance in value:
                knn_instances.append(instance)         
                knn_scores.append(key)
        
        # Get the predicition from voting function
        maj_label_list = vote_fun(knn_instances, knn_scores, label_col)
        predict_label = maj_label_list[0]

        # Update confusion matrix row with prediction
        conf_matrix[cur_row_index][predict_label] = conf_matrix[cur_row_index][predict_label] + 1

    return conf_matrix



def accuracy(confusion_matrix, label):
    """Returns the accuracy for the given label from the confusion matrix.
    
    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the accuracy of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Calculate the true negatives for label
    true_negatives = 0

    for i in range(confusion_matrix.row_count()):

        # Sum values that are true negatives
        if row_label_idx != i:

            # Traverse each row and add values that are not actual counts or the label
            for attribute in col_list:        
                if attribute != label and attribute != 'actual':
                    true_negatives = true_negatives + confusion_matrix[i][attribute]

    # Calculate the number of predicted instances
    instance_counts = [0 for i in range(confusion_matrix.row_count())]

    for row in confusion_matrix:
        for label in col_list:
            if label != 'actual':
                instance_counts[col_list.index(label)-1] = instance_counts[col_list.index(label)-1] + row[label]

    instance_sum = sum(instance_counts)

    # Determine accuracy with variables found
    label_accuracy = (true_positives + true_negatives) / instance_sum
    return label_accuracy



def precision(confusion_matrix, label):
    """Returns the precision for the given label from the confusion
    matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the precision of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Determine the total predicted calculation
    total_predicted = 0
    for row in confusion_matrix:
        total_predicted = total_predicted + row[label]

    # Check if both are 0 (100% guess given zero instances):
    if total_predicted == 0 and true_positives == 0:
        return 1
    
    # Determine precision with variables
    return (true_positives) / (total_predicted)



def recall(confusion_matrix, label): 
    """Returns the recall for the given label from the confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.
        label: The class label to measure the recall of.

    """
    # Get the row index of label (take into account 'actual' column)
    col_list = confusion_matrix.columns()
    row_label_idx = col_list.index(label) - 1

    # Determine the true positive calculation
    true_positives = confusion_matrix[row_label_idx][label]

    # Determine the total actual labels
    total_positives = 0
    for i in range(1,len(col_list)):
        total_positives = total_positives + confusion_matrix[row_label_idx][col_list[i]]
    
    # Determine recall with variables
    return true_positives / total_positives