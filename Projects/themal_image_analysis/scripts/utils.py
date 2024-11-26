import os

def create_directories(directories):
    """
    Create directories if they do not exist.
    
    Args:
    - directories (list): List of directory paths to create.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

if __name__ == "__main__":
    directories = ['./data/train', './data/test', './models', './results']
    create_directories(directories)
