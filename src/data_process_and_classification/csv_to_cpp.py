import os

def convert_csv_to_cpp(input_file: str, array_name: str):
    """Convert CSV (skipping header) into a C++ array declaration file.
    Output is saved in the same folder as this Python script.
    """
    try:
        with open(input_file, "r") as fin:
            # Skip header line
            next(fin)
            data = []
            for line in fin:
                row = []
                for cell in line.strip().split(","):
                    try:
                        row.append(float(cell))
                    except ValueError:
                        print(f"Warning: Skipping non-numeric cell: {cell}")
                if row:
                    data.append(row)
    except IOError:
        print(f"Error opening input file: {input_file}")
        return None

    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0

    # Always save output in the same folder as the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(script_dir, f"{base_name}_array.cpp")

    try:
        with open(output_file, "w") as fout:
            if cols > 1:  # 2D array
                fout.write(f"float {array_name}[{rows}][{cols}] = {{\n")
                for i, row in enumerate(data):
                    fout.write("    {" + ", ".join(str(x) for x in row) + "}")
                    if i < rows - 1:
                        fout.write(",")
                    fout.write("\n")
                fout.write("};\n")
            else:  # 1D array
                fout.write(f"float {array_name}[{rows}] = {{")
                fout.write(", ".join(str(row[0]) for row in data))
                fout.write("};\n")
    except IOError:
        print(f"Error writing output file: {output_file}")
        return None

    print(f"✅ Converted {rows} rows and {cols} columns into array '{array_name}' and saved to {output_file}")
    return output_file


# Example usage:
if __name__ == "__main__":
    # Change these paths/names to your actual files
    convert_csv_to_cpp("./4_FEATS_COMBINED/9.71_Hz_sampling/X_test_9.71.csv", "X_test")
    convert_csv_to_cpp("./4_FEATS_COMBINED/9.71_Hz_sampling/y_test_9.71.csv", "y_test")

