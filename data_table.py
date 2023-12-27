"""
HW-5 Data Table implementation.

NAME: David Giacobbi
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """
        # Check if column input is valid
        if column not in self.columns():
            raise IndexError('bad column name')

        # Delete list elements from column provided    
        del_index = self.columns().index(column)
        del self.__columns[del_index]
        del self.__values[del_index]

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        # Check if the DataRows have equal length
        if len(self.__columns) != len(other.columns()):
            return False
        
        # Check if each element in the DataRow are equivalent to each other
        for name in self.__columns:
            # If DataRows do not have same column names, throw exception and return false
            try:
                if self[name] != other[name] or self[name] != other[name]:
                    return False
            except:
                return False
        
        # If all other tests work return true
        return True

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns]


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
        # Return complete list if no columns specified
        if columns == None:
            return DataRow(self.columns(), self.values())
        
        else:
            # Return new data row with specific columns and corresponding values
            return DataRow(columns, self.values(columns))


    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        # Create a list of row data values
        data_table = []
        for row in range(self.row_count()):
            data_table.append(self.__row_data[row].values())

        return tabulate.tabulate(data_table, headers=self.columns())

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        # Check if row index input is valid
        if row_index not in range(self.row_count()):
            raise IndexError('bad row index')

        # Delete list elements from row index provided
        del self.__row_data[row_index]
        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        # Check to see if row_values list is correct size
        if len(row_values) == self.column_count():
            # Create a new DataRow object and add to the row_data
            self.__row_data.append(DataRow(self.columns(), row_values))
        else:
            raise ValueError

    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        # Create a new table to return
        new_table = DataTable(self.columns())

        # Traverse index list
        for idx in row_indexes:
            # Check if index is valid and then append corresponding values
            if idx not in range(self.row_count()):
                raise IndexError
            else:
                new_row = self.__row_data[idx]
                new_table.append(new_row.values())
        
        # Return appended table
        return new_table

    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        # Hard update value given parameters
        self.__row_data[row_index][column] = new_value


    def drop(self, columns):
        """Removes the given columns from the current table.

        Args:
            column: the name of the columns to drop
        """
        # Drop DataTable columns
        drop_cols = self.columns()
        for col in columns:
            drop_cols.remove(col)

        # Append each row from old table into drop table
        drop_table = DataTable(drop_cols)
        for row in self:
            new_row = row.select(drop_cols)
            drop_table.append(new_row.values())
        
        self.__columns = drop_table.columns()
        self.__row_data = drop_table.__row_data


    
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine are must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """
        # Create Combined Column List
        table2_combine = table2.columns()
        for attribute in columns:
            
            # Try to remove columns from table2
            try:
                table2_combine.remove(attribute)
            except:
                raise IndexError
            
            # Check if table1 has column values
            if attribute not in table1.columns():
                raise IndexError
            
        combine_col = table1.columns() + table2_combine

        # Create an Empty Table for Combined Values
        combine_table = DataTable(combine_col)
        row_added_val = []

        # If there are no columns to combine on
        if len(columns) == 0:
            combine_table = DataTable(table1.columns() + table2.columns())

            # Fill table with row1 and row2 values concatenated
            for row1 in table1:
                for row2 in table2:
                    combine_table.append(row1.values() + row2.values())

            return combine_table

        # Check each row in table2 to table1 for matches
        for row1 in table1:
            attribute1 = row1.select(columns)
            row1_matched = False

            # Inner Join Check
            for row2 in table2:
                attribute2 = row2.select(columns)

                # Compare attribute lists from table 1 and 2
                if attribute1 == attribute2:
                    # Add to the combine data table with joined values
                    combine_table.append(row1.values() + row2.values(table2_combine))
                    # Match found
                    row1_matched = True
                    row_added_val.append(row2.values(columns))
                
            # Outer Join Condition
            if non_matches and not row1_matched:
                # Fill row with empty values for row2
                nan_row = ['' for val in range(len(table2_combine))]
                # Add to the combine data table with joined values
                combine_table.append(row1.values() + nan_row)
        
        # Outer Join for non-matches in table 1
        if non_matches:

            for row2 in range(table2.row_count()):
                
                # Find values that do not have a match in table1 or added yet
                if table2[row2].values(columns) not in row_added_val:

                    # Create a new row and fill in known table 2 values
                    curr = DataRow(combine_col, ['' for x in combine_col])
                    for col in table2[row2].columns():
                        curr[col] = table2[row2][col]

                    combine_table.append(curr.values())

        return combine_table

    
    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        # Try if string has integer value
        try:
            int_val = int(value)
            return int_val

        except:
            # If integer fails, try float value
            try:
                float_val = float(value)
                return float_val
            
            except:
                # Return value that cannot be converted to int or float
                return value