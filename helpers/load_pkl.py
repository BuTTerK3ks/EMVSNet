import pickle

path = "/home/grannemann/PycharmProjects/AA-RMVSNet/outputs_dtu/dropout/batch_9.pkl"

with open(path, "rb") as f:
    obj = pickle.load(f)

test1 = obj['image_outputs_list'][5]['alea_1'].squeeze()
test1_norm = (test1 - test1.min()) / (test1.max() - test1.min())

print(obj)