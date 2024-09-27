def create_data_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Make sure the line is in the correct format
            assert len(parts) == 3
            code = parts[0]
            name = parts[2]
            data_dict[code] = name
    return data_dict

file_path = './data_dictionary.txt'

data_dict = create_data_dict(file_path)

