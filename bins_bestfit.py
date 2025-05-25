import time 
import matplotlib.pyplot as plt 
from scipy.stats import linregress

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

# Example usage:
problem_info = {
    'id': "9",
    'bin_capacity': 14,
    'num_items': 9,
    'num_bins': 6,
    'bin_capacities': [5, 7, 3, 5, 12, 11, 10, 11, 9]
}

result = best_fit(problem_info)
print(result)

def read_bin_packing_file(file_path):
    with open(file_path, 'r') as file:
        num_problems = int(file.readline().strip())

        problems = []
        for _ in range(num_problems):
            #problem identifier 
            problem_id = file.readline().strip()
            #bin capacity, number of items, and number of bins in best sol.
            bin_capacity, num_items, num_bins = map(int, file.readline().split())

            # individual bin capacities (len = n)
            bin_capacities = [int(file.readline().strip()) for _ in range(num_items)]

            # Store the problem information in a dictionary
            problem_info = {
                'id': problem_id,
                'bin_capacity': bin_capacity,
                'num_items': num_items,
                'num_bins': num_bins,
                'bin_capacities': bin_capacities
            }

            problems.append(problem_info)
    return problems

def main():
    total_times = {}
    for i in range(1,5):
        file_path = 'binpack'+str(i)+'.txt'
        problems = read_bin_packing_file(file_path)

        start_all = time.time()
        for ind, problem_info in enumerate(problems):
            size = problem_info['num_items']
            start_prob = time.time()
            bins_used = best_fit(problem_info)
            end_prob = time.time()
            runtime = end_prob - start_prob
            if size not in total_times:
                total_times[size] = runtime
            else:
                prev = total_times[size]
                total_times[size] = (prev+runtime)/2

            print(f"{ind}: {bins_used == problem_info['num_bins']}, Time used: {end_prob - start_prob:.3g} s, number of items: {problem_info['num_items']}, bins: {bins_used}")

        total_t = time.time() - start_all
        print(f"Total time: {total_t:.3g}s")

    items = list(total_times.keys())
    runtimes = list(total_times.values())

    # Create a bar plot
    plt.bar(items, runtimes, color='blue', alpha=0.7, label='Runtime')

    # Calculate the regression line
    slope, intercept, _, _, _ = linregress(items, runtimes)
    regression_line = [slope * item + intercept for item in items]

    # Plot the regression line
    plt.plot(items, regression_line, color='red', label='Regression Line')

    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Items with Regression Line')
    plt.legend()
    plt.show()

main()