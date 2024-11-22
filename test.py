import os 
import logging 

network_metadata = [
    {
        "net":"",
        "path": "database/network_4.py",
        "prompt": "Add convolutional layer to improve the above network, can train on Cifar10. Only output the class definition with its methods.",
        "score": 1345.223
    },
    {
        "net":"",
        "path": "database/network_6.py",
        "prompt": "Add residual connection to improve the above network, can train on Cifar10. Only output the class definition with its methods.",
        "score": 876.122
    },
    {
        "net":"",
        "path": "database/network_2.py",
        "prompt": "improve the above network by reducing the size drastically, can train on Cifar10. Only output the class definition with its methods.",
        "score": 815.765
    },
    {
        "net":"",
        "path": "database/network_3.py",
        "prompt": "Add a layer to improve the above network, can train on Cifar10. Only output the class definition with its methods.",
        "score": 762.412
    },
    {
        "net":"",
        "path": "database/network_5.py",
        "prompt": "Add recurrent layer to improve the above network, can train on Cifar10. Only output the class definition with its methods.",
        "score": 634.565
    }
]

def selection(population, max_pop_size):
    kept_metadata = population[:max_pop_size]

    removed_metadata = network_metadata[max_pop_size: ]
    for item in removed_metadata:
        file_path = item['path']
        if os.path.exists(file_path):
            try: 
                os.remove(file_path)
            except Exception as e: 
                logging.error(f"Error deleting file {file_path}: {e}")
        else: 
            logging.warning(f"File not found: {file_path}")     

    return kept_metadata

processed_data = selection(network_metadata, 3)
print(0)
