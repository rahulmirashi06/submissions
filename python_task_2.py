import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    distance_matrix.values[[range(len(distance_matrix))] * 2] = 0

    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']

        distance_matrix.at[start_id, end_id] = distance
        distance_matrix.at[end_id, start_id] = distance

    distance_matrix = distance_matrix.fillna(0).cumsum(axis=1).cumsum(axis=0)

    return distance_matrix


# Example usage:
dataset_path = 'C:\\Users\\acer\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv'
df = pd.read_csv(dataset_path)
distance_matrix = calculate_distance_matrix(df)
print("Distance Matrix:")
print(distance_matrix)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []

    for i in range(len(df.index)):
        for j in range(i + 1, len(df.columns)):
            id_start = df.index[i]
            id_end = df.columns[j]
            distance = df.at[id_start, id_end]

            unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


# Example usage:
unrolled_df = unroll_distance_matrix(distance_matrix)
print("Unrolled Distance Matrix:")
print(unrolled_df)



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    threshold = 0.1 * reference_avg_distance

    result_df = df.groupby('id_start')['distance'].mean().reset_index()
    result_df = result_df[(result_df['distance'] >= reference_avg_distance - threshold) &
                          (result_df['distance'] <= reference_avg_distance + threshold)]

    return result_df


# Example usage:
reference_id = 1  # Replace this with the desired reference ID
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print("IDs within 10% threshold of the reference ID's average distance:")
print(result_df)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df


# Example usage:
result_df = calculate_toll_rate(unrolled_df)
print("DataFrame with Toll Rates:")
print(result_df)



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    weekday_time_ranges = [(datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
                           (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
                           (datetime.time(18, 0, 0), datetime.time(23, 59, 59))]

    weekend_time_ranges = [(datetime.time(0, 0, 0), datetime.time(23, 59, 59))]

    df['start_day'] = df['start_datetime'].dt.strftime('%A')
    df['start_time'] = df['start_datetime'].dt.time
    df['end_day'] = df['end_datetime'].dt.strftime('%A')
    df['end_time'] = df['end_datetime'].dt.time

    # Function to calculate discount factor based on time range
    def calculate_discount_factor(row):
        if row['start_day'] in ['Saturday', 'Sunday']:
            return 0.7  # Constant discount factor for weekends
        else:
            for start_range, end_range in weekday_time_ranges:
                if start_range <= row['start_time'] <= end_range:
                    return 0.8 if start_range <= row['end_time'] <= end_range else 1.2
            return 1.0  # Default if no match

    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
        df[vehicle_type] = df.apply(lambda row: row[vehicle_type] * calculate_discount_factor(row), axis=1)

    return df


# Example usage:
unrolled_df = unroll_distance_matrix(distance_matrix)
result_df = calculate_time_based_toll_rates(unrolled_df)
print("DataFrame with Time-Based Toll Rates:")
print(result_df)
