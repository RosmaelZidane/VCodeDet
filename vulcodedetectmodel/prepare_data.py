import data_preprocessing as dpre  ### Always usde this fucntion if you have to use a function from data_preprocessing.py


"""This code allows to obtain all the meta data informations"""

def prepared():
    dpre.datasetss()
    dpre.get_dep_add_lines_datasetss()
    return 

if __name__ == "__main__":
    prepared()  