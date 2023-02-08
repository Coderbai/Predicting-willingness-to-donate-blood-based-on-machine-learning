from utils import *
import pandas as pd


if __name__ == '__main__':
    label_name = ['./data/message_data/yangzhou/data.xls']
    time_ = [pd.Timestamp('2015-10-16'), pd.Timestamp('2016-10-10'), pd.Timestamp('2019-9-6'), pd.Timestamp(pd.datetime.now())]
    # file_name = [('./data/raw_data/yangzhou/2008-2016.xls', 'SQL Results (2)'),
    #              ('./data/raw_data/yangzhou/2016-2019.xls', 'SQL Results'),
    #              ('./data/raw_data/yangzhou/2016-2019.xls', 'SQL Results (1)')]
    file_name = [('./data/raw_data/yangzhou2/2016-2019.xls', 'SQL Results'),
                 ('./data/raw_data/yangzhou2/2016-2019.xls', 'SQL Results (1)'),
                 ('./data/raw_data/yangzhou2/2016-2019.xls', 'SQL Results (2)')]
    data = []
    label = []
    id_ = []
    time_sv = []
    for file, sheet in file_name:
        data.append(read_file(file, sheet))
    print(len(data)) # 3
    for file in label_name:
        xls_file = pd.ExcelFile(file)
        for sheet_name in xls_file.sheet_names:
            a = pd.read_excel(file, sheet_name=sheet_name, converters={u'识别码': str})
            print(len(a)) # 65535
            # print(a)
            b = list(zip(a.iloc[:, 1], a.iloc[:, 4]))
            # assert len(a.iloc[:, 4]) == len(set(a.iloc[:, 4]))
            id_ = id_ + list(a.iloc[:, 4])
            time_sv = time_sv + list(a.iloc[:, 1])
            print(len(b)) #65535
            # print(b)
            for date, id in b:
                # print('-----------')
                # print(date,id)
                date_ = pd.Timestamp(date)
                # print(date_)
                add_label = False
                for x in data:
                    cur_data = list(x.iloc[:, 1])
                    # print('cur_data',cur_data,cur_data.count(id))
                    if cur_data.count(id) == 0:
                        continue
                    try:
                        j = cur_data.index(id)
                    except BaseException:
                        continue
                    # print('j',j)
                    while True:
                        if j > len(cur_data)-1 or cur_data[j] != id:
                            break
                        # print('(x.iloc[j, 6]-date_).days',(x.iloc[j, 6]-date_).days,x.iloc[j, 6],date_)
                        if (x.iloc[j, 6]-date_).days <= 7 and (x.iloc[j, 6]-date_).days >= 0:
                            add_label = True
                            label.append(1)
                            break
                        j += 1
                    # if j > 65535:
                    #     continue
                    # else:
                    #     break
                if not add_label:
                    label.append(0)
                # for i, top in enumerate(time_):
                #     if date_ < top:
                #         add_label = False
                #         cur_data = list(data[i-1].iloc[:, 1])
                #         if cur_data.count(id) == 0:
                #             label.append(0)
                #             break
                #         index_ = [i for i in range(len(cur_data)) if cur_data[i] == id]
                #         for j in index_:
                #             if (data[i-1].iloc[j, 6]-date_).days <= 7:
                #                 label.append(1)
                #                 add_label = True
                #                 break
                #         if not add_label:
                #             label.append(0)
                #         break
    label = np.array(label)
    id = np.array(id_)
    time_sv = np.array(time_sv)
    print(len(label))
    print(len(id))
    print((len(time_sv)))
    np.savez('./data/exp_data/yangzhou2/label/label1.npz', id_, label, time_sv)

    # # print(len(label))
    # a = np.load('./label.npz', allow_pickle=True)
    # print(a.files)
    # print(a['arr_0'])
    # print(a['arr_1'])
    # print(a['arr_2'])
    # label_rate = a['arr_1'].sum() / len(a['arr_1'])
    # print(label_rate)
