import numpy as np

def ratio_scaling(ratio):
    return (np.log2(ratio)+5)/10

def T_scaling(T):
    return T/400.0 - 1.5

def P_scaling(P):
    return (np.log10(P)-2)/5

class DataProcessing:
    def __init__(self, origin_filename=""):
        self.data_filename = origin_filename
        self.input_data = np.load(self.raw_data_path + "/" + self.data_filename)
        pass

    input_data = None
    data_filename = ""
    raw_data_path = "./data/raw"
    output_path = "./data/output"

    def set_origindata(self, filename):
        self.data_filename = filename
        return True
    
    def split_data_randomly(self, percentage):
        len_data = len(self.input_data)
        len_train = int(len_data * percentage)
        temp_data = self.input_data.copy()
        train_input = temp_data[:len_train, 1:]
        train_output = temp_data[:len_train, 0]
        valid_input = temp_data[len_train:, 1:]
        valid_output = temp_data[len_train:, 0]
        return train_input, train_output, valid_input, valid_output

    def split_data(self, percentage):
        len_data = len(self.input_data)
        len_train = int(len_data * percentage)
        train_input = self.input_data[:len_train, 1:]
        train_output = self.input_data[:len_train, 0]
        valid_input = self.input_data[len_train:, 1:]
        valid_output = self.input_data[len_train:, 0]
        return train_input, train_output, valid_input, valid_output

    def normalize_data(self, inputdata):
        normalize = np.zeros(shape = inputdata.shape)
        normalize[:,0] = T_scaling(inputdata[:, 0])
        normalize[:,1] = P_scaling(inputdata[:, 1])
        normalize[:,2] = ratio_scaling(inputdata[:, 2])
        normalize[:,3] = ratio_scaling(inputdata[:, 3])
        return normalize

    def auto_processing(self, percentage=0.7):
        ti, to, vi, vo = self.split_data_randomly(percentage)
        ti = self.normalize_data(ti)
        vi = self.normalize_data(vi)
        return ti, to, vi, vo


#dp = DataProcessing("database_clf_large_800T.npy")
#ti, _, _, _ = dp.auto_processing()
#print(ti)