import os

def create_directories(categories):
    for category in categories:
        if not os.path.exists(category):
            os.makedirs(category)
