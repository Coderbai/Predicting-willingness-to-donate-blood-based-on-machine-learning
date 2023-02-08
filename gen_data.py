#generate data is ok

from utils import *
import pandas as pd
from collections import OrderedDict, Counter
import time
import os
import numpy as np

if __name__ == '__main__':
    files = os.listdir('./data/exp_data_xlsx/yangzhou2/')
    files.sort()
    a = np.load('./data/exp_data/yangzhou2/label/label.npz', allow_pickle=True)
    id_, label, time_sv = a['arr_0'], a['arr_1'], a['arr_2']
    print(files)
    print('label',label)
    print(Counter(np.array(label)))
    print(label.sum()/len(label))
    time_sv = [times[:10] for times in time_sv]
    pos = 0
    sv_dict = OrderedDict()
    label_ = []
    key_lost = []
    for file in files:
        if '.xlsx' not in file:
            continue
        print('file[:10]',file[:10])
        print(time_sv.count(file[:10]))
        sv_range = time_sv.count(file[:10]) + pos
        result = data_recovery('./data/exp_data_xlsx/yangzhou2/'+file, sheet='sheet1')
        # print('result',result)
        id_test = set(id_[pos:sv_range])
        print(len(id_test))
        date_error = 0
        for u in range(pos, sv_range):
            if id_[u] in result.keys():
                if id_[u] in sv_dict.keys():
                    # print(id_[u])
                    print('key exists')
                    continue
                sv_dict[id_[u]] = result.get(id_[u])
                label_.append(label[u])
            else:
                # print('date ERROR')
                # date_error += 1
                key_lost.append(id_[u])
                pass
        print('Num of people not in blood data set:', sv_range-pos-len(label_))
        npdata = read_data_from_dict(sv_dict, file[:10])
        assert len(npdata) == len(label_)
        np.savez('./data/exp_data/yangzhou2/data_nos/'+file[:10]+'.npz', npdata, label_)
        pos = sv_range
        sv_dict.clear()
        label_.clear()
        del result

    # key_lost = list(set(key_lost))
    # writer = pd.ExcelWriter('./a.xlsx')
    # key_lost_ = defaultdict(list)
    # key_lost_['识别码'] = key_lost
    # key_lost_ = pd.DataFrame(key_lost_).to_excel(writer, sheet_name='sheet')
    # writer.save()