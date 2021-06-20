import argparse

def get_input_args_train():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'Path to flower images'" folder")
    parser.add_argument('--save_dir', type = str, default = 'checkpoint/', help = 'Path to save checkpoint files')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'CNN Architecture Model')
    parser.add_argument('--learning_rate', type = str, default = '0.003', help = 'Hyperparameter - Learning rate')
    parser.add_argument('--hidden_units', type = str, default = "1000", help = 'Parameters - Hidden Units')
    parser.add_argument('--eppochs', type = str, default = '3', help = 'Parameters - Eppochs')
    parser.add_argument('--device', type = str, default = 'GPU', help = 'Device type')
    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    
    in_args_train = parser.parse_args()
    
    return in_args_train