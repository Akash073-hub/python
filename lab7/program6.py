def merge_dictionaries():
    try:
        dict1_str = input("Enter the first dictionary (in Python dictionary format, e.g., {'a': 1, 'b': 2}): ")
        dict1 = eval(dict1_str)  
        dict2_str = input("Enter the second dictionary (in Python dictionary format, e.g., {'c': 3, 'd': 4}): ")
        dict2 = eval(dict2_str) 
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            raise ValueError("Input must be valid Python dictionaries.")
        merged_dict = dict1.copy()  
        for key, value in dict2.items():
            merged_dict[key] = value  
        print("Merged dictionary:", merged_dict)
    except (SyntaxError, NameError, ValueError) as e:
        print(f"Error: Invalid dictionary input. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
merge_dictionaries()