def best_fit(problem_info):
    # Extract information from the problem_info dictionary
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    num_bins = problem_info['num_bins']
    bin_capacities = problem_info['bin_capacities']

    # Initialize the bins with their capacities
    bins = [{'id': i, 'capacity': bin_capacities[i], 'items': []} for i in range(num_bins)]

    # Iterate through each item and assign it to the best-fitting bin
    for item_id in range(num_items):
        item_size = bin_capacities[item_id]

        # Find the bin with the best fit for the current item
        best_fit_bin = None
        min_remaining_capacity = float('inf')

        for bin_id in range(num_bins):
            remaining_capacity = bins[bin_id]['capacity'] - item_size

            if remaining_capacity >= 0 and remaining_capacity < min_remaining_capacity:
                best_fit_bin = bin_id
                min_remaining_capacity = remaining_capacity

        # If a suitable bin is found, add the item to it
        if best_fit_bin is not None:
            bins[best_fit_bin]['items'].append(item_id)
            bins[best_fit_bin]['capacity'] -= item_size

    return len(bins)
