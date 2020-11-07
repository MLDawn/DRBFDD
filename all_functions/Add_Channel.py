import numpy as np
import torch

def Add_Channel(batch_size, data):
    l = []
    for i in range(batch_size):
        s = data[i].numpy()
        cat = np.stack((s,) * 3, axis=-1)  # Resulsts in (28,28,3)
        cat = np.reshape(cat, (3, 28, 28))
        l.append(cat)
    data = torch.tensor(np.array(l))
    return data