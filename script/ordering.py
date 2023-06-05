import sys

# Get the input file name from the command line
if len(sys.argv) < 2:
    print("Error: Please specify the input file name.")
    sys.exit(1)

input_file = sys.argv[1]

# Read the content of the input file
try:
    with open(input_file, 'r') as file:
        lines = file.readlines()
except IOError:
    print(f"Error: Unable to open the file '{input_file}'.")
    sys.exit(1)

# Split each line into columns and convert the third column to float
lines = [line.strip().split() for line in lines]
lines = [(line[0], line[1], float(line[2])) for line in lines]

# Sort the lines based on the value of the third column
lines.sort(key=lambda x: x[2])

# Create a name for the output file based on the input file name
output_file = f'{input_file}'

# Write the sorted lines to a new file
try:
    with open(output_file, 'w') as file:
        file.write("num_blocks_factor num_threads avg_epoch_training_time\n")
        for line in lines:
            file.write(f'{line[0]} {line[1]} {line[2]}\n')
except IOError:
    print(f"Error: Unable to write to the file '{output_file}'.")
    sys.exit(1)

print(f"The lines have been reordered and saved in the file '{output_file}'.")
