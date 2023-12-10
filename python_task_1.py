import pandas as pd
import numpy as np


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    return car_matrix


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv'
df = pd.read_csv(dataset_path)
result_matrix = generate_car_matrix(df)
print("Task 1 Question 1")
print(result_matrix)
print("--------------")

def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]

    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices, default='unknown'))

    type_counts = df['car_type'].value_counts().to_dict()

    type_counts = dict(sorted(type_counts.items()))

    return type_counts


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv'
df = pd.read_csv(dataset_path)
result_type_counts = get_type_count(df)
print("Task 1 Question 2")
print(result_type_counts)
print("--------------")

def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_mean = df['bus'].mean()

    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    return sorted(bus_indexes)

# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv'
df = pd.read_csv(dataset_path)
result_bus_indexes = get_bus_indexes(df)
print("Task 1 Question 3")
print(result_bus_indexes)
print("--------------")


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_avg_truck = df.groupby('route')['truck'].mean()

    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    filtered_routes.sort()

    return filtered_routes


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv'
df = pd.read_csv(dataset_path)
result_filtered_routes = filter_routes(df)
print("Task 1 Question 4")
print(result_filtered_routes)
print("--------------")


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modified_matrix = matrix.copy()

    modified_matrix[matrix > 20] *= 0.75
    modified_matrix[(matrix <= 20) & (matrix > 0)] *= 1.25

    modified_matrix = modified_matrix.round(1)

    return modified_matrix


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv'
df = pd.read_csv(dataset_path)

result_matrix = generate_car_matrix(df)

modified_result_matrix = multiply_matrix(result_matrix)
print("Task 1 Question 5")
print(modified_result_matrix)
print("--------------")



def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    df = df.dropna(subset=['start_datetime', 'end_datetime'])

    df['duration'] = df['end_datetime'] - df['start_datetime']

    result_series = df.groupby(['id', 'id_2']).apply(check_time_coverage)

    return result_series


def check_time_coverage(group):
    days_covered = group['start_datetime'].dt.day_name().nunique() == 7

    hours_covered = (group['end_datetime'].max() - group['start_datetime'].min()).total_seconds() >= 24 * 60 * 60

    return days_covered and hours_covered


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-2.csv'
df = pd.read_csv(dataset_path)
result_series = time_check(df)
print("Task 1 Question 6")
print(result_series)
print("--------------")
