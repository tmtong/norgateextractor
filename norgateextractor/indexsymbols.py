import json
import os

# Specify the directory containing the JSON files
json_directory = 'data/index/'

# Iterate over each JSON file in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)
        # Extract the index name without the .json extension
        index_name = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Initialize a set to store unique symbols for this index
                symbols = set()
                
                # Extract symbols from each date entry
                for date, constituents in data.items():
                    if isinstance(constituents, list):
                        symbols.update(constituents)
                
                # Convert the set to a sorted list
                sorted_symbols = sorted(symbols)
                
                # Save the symbols to a text file with .symbol extension
                output_file = f'{index_name}.symbol'
                output_path = os.path.join(json_directory, output_file)
                with open(output_path, 'w') as outfile:
                    for symbol in sorted_symbols:
                        outfile.write(f"{symbol}\n")
                
                print(f"Symbols for {index_name} have been saved to {output_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

