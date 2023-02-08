import pandas as pd
from collections import defaultdict
import numpy as np
import copy


# raw_col = [' 0 ', '1        ', '2    ', '3      ', '4      ', '5   ', '6       ', '7      ', '8  ',
#            '9   ', '10 ', '11    ', '12   ', '13 ', '14  ', '15     ', '16 ', '17     ', '18     ',
#            '19  ', '20      ', '21     ', '22     ', '23     ', '24     ', '25          ']
raw_col = ['   ', '献血者识别码', '序列号', '证件类型', '登记时间', '献血量', '采血时间', '献血地点', '血型',
           'Rh血型', '性别', '出生日期', '国籍', '民族', '年龄', '居住类型', '职业', '文化程度', '所属区县',
           '工作组', '实际采血量', '检验结论', '献血方式', '采血类型', '有非标量', '是否有献血反应']

# data_col = ['0     ',  '1     ', '2   ', '3  ',  '4  ', '5              ', '6           ', '7      ', '8       '
#             '9           ', '10     ', '11 ' , '12        ', '13        ', '14          ', '15         ',
#             '16   ', '17   ', '18          ']
data_col = ['身份识别码', '出生年份', '性别', '民族', '血型', '最近一次献血时间', '最近一次献血量', '总献血量', '献血次数',
            '最近献血是否合格', '献血频率', '职业', '初次输血时间', '多长时间未献血', '上次献血地点', '文化程度',
            'Rh血型', '居住类型', '是否有献血反应']

blood_type = ['A', 'B', 'AB', 'O']

key_col = '身份识别码'

def read_file(file, name):
    return pd.read_excel(file, sheet_name=name, converters={u'献血者识别码': str, u'序列号': str, u'身份识别码': str})

def isnan(inputs):
    try:
        rel = np.isnan(inputs)
    except TypeError:
        return False
    else:
        return rel


def data_match(xls_data, pre_data=None, now_time='now', pre_time='1970'):
    '''
    :param xls_data: pandas.DataFrame
    :param pre_data: dict
    :param now_time: 'yyyy-mm-dd HH:MM:ss' or default is current time
    :return: dict or defaultdict
    '''
    assert isinstance(xls_data, pd.DataFrame)
    data_ = defaultdict(list) if pre_data is None else pre_data
    now_time = pd.Timestamp(pd.datetime.now()) if now_time == 'now' else pd.Timestamp(now_time)
    pre_time = pd.Timestamp(pd.datetime(1970, 1, 1)) if pre_time == '1970' else pd.Timestamp(pre_time)

    # process excel data to dict
    for i in xls_data.index.values:
        key_data = xls_data.iloc[i, 1]
        # collect blood time is later than current time, continue
        # register time is not equal to collect blood time, continue
        try:
            if isinstance(xls_data.iloc[i, 6], str):
                xls_data.iloc[i, 4] = pd.Timestamp(xls_data.iloc[i, 4])
                xls_data.iloc[i, 6] = pd.Timestamp(xls_data.iloc[i, 6])
                xls_data.iloc[i, 11] = pd.Timestamp(xls_data.iloc[i, 11])
            if xls_data.iloc[i, 6] <= pre_time or xls_data.iloc[i, 6] > now_time:
                continue
            # if xls_data.iloc[i, 4] > xls_data.iloc[i, 6]:
            #     continue
            if '单' in xls_data.iloc[i, 23]:
                continue
        except BaseException:
            continue

        if key_data in data_.keys():
            pdata = data_[xls_data.iloc[i, 1]]
            if pdata[4] != xls_data.iloc[i, 8] and not isnan(pdata[4]) and not isnan(xls_data.iloc[i, 8]):
                print('blood type not consistant, Please check!', pdata[0])
                if xls_data.iloc[i, 6] > pdata[12]:
                    pdata[4] = xls_data.iloc[i, 8]
            if (isnan(pdata[1]) or not isinstance(pdata[1], pd.Timestamp)) and not isnan(xls_data.iloc[i, 11]):
                try:
                    pdata[1] = xls_data.iloc[i, 11]
                except BaseException:
                    pass
            pdata[4] = xls_data.iloc[i, 8] if isnan(pdata[4]) else pdata[4]
            pdata[5] = xls_data.iloc[i, 6] if xls_data.iloc[i, 6] > pdata[5] else pdata[5]  # guarantee values is max
            if isinstance(xls_data.iloc[i, 20], str):
                num_blood = xls_data.iloc[i, 20]
                blood_ = 0
                if ',' in num_blood:
                    all_num = num_blood.split(',')
                    for k in all_num:
                        blood_ = blood_ + int(k)
                elif 'ml' in num_blood:
                    blood_ = int(num_blood[:-2])
                else:
                    blood_ = int(num_blood)
            else:
                blood_ = xls_data.iloc[i, 20]
            pdata[6] = blood_
            pdata[7] += pdata[6]
            pdata[8] += 1
            pdata[9] = xls_data.iloc[i, 21]
            pdata[12] = xls_data.iloc[i, 6] if xls_data.iloc[i, 6] < pdata[12] else pdata[12]  # guarantee values is min
            pdata[10] = (pdata[5] - pdata[12]) / (pdata[8] - 1)
            if isnan(pdata[11]) or pdata[11] != xls_data.iloc[i, 16]:
                pdata[11] = xls_data.iloc[i, 16]
            pdata[14] = xls_data.iloc[i, 18]
            if isnan(pdata[15]) or pdata[15] != xls_data.iloc[i, 17]:
                pdata[15] = xls_data.iloc[i, 17]
            if isnan(pdata[17]) or pdata[17] != xls_data.iloc[i, 15]:
                pdata[17] = xls_data.iloc[i, 15]
            pdata[18] = xls_data.iloc[i, 25]
        else:
            # database years extend the pandas max
            try:
                xls_data.iloc[i, 11]
                xls_data.iloc[i, 20]
            except BaseException:
                continue
            if isinstance(xls_data.iloc[i, 20], str):
                num_blood = xls_data.iloc[i, 20]
                blood_ = 0
                if ',' in num_blood:
                    all_num = num_blood.split(',')
                    for k in all_num:
                        blood_ = blood_ + int(k)
                elif 'ml' in num_blood:
                    blood_ = int(num_blood[:-2])
                else:
                    blood_ = int(num_blood)
            else:
                blood_ = xls_data.iloc[i, 20]
            pdata = [xls_data.iloc[i, 1], xls_data.iloc[i, 11], xls_data.iloc[i, 10], xls_data.iloc[i, 13], xls_data.iloc[i, 8], xls_data.iloc[i, 6], blood_, blood_, 1,
                     xls_data.iloc[i, 21], 0, xls_data.iloc[i, 16], xls_data.iloc[i, 6], 0, xls_data.iloc[i, 18], xls_data.iloc[i, 17],
                     xls_data.iloc[i, 9], xls_data.iloc[i, 15], xls_data.iloc[i, 25]]
            data_[key_data] = pdata
    return data_


def data_clip(inputs, now_time='now'):
    '''
    :param inputs: dict
    :param now_time: 'yyyy-mm-dd HH:MM:ss' or default is current time
    :return: dict or defaultdict
    '''
    if now_time == 'now':
        now_time = pd.Timestamp(pd.datetime.now())
    else:
        now_time = pd.Timestamp(now_time)
    # if type of blood is lack, this data should not be used
    pop_list = []
    for key in inputs.keys():
        if isnan(inputs[key][4]) or isnan(inputs[key][9]) or \
                isnan(inputs[key][1]) or not isinstance(inputs[key][1], pd.Timestamp):
            pop_list.append(key)
            continue
        # revalue 多长时间未献血 depended on now time
        inputs[key][13] = now_time - inputs[key][5]
    for key in pop_list:
        inputs.pop(key)
    return inputs


def classify_data_by_blood(inputs):
    assert isinstance(inputs, defaultdict) or isinstance(inputs, dict)
    blood_dict = {'A': defaultdict(list), 'B': defaultdict(list), 'AB': defaultdict(list), 'O': defaultdict(list)}

    for key in inputs.keys():
        if inputs[key][4] == blood_type[0]:
            blood_dict[blood_type[0]][key] = inputs[key]
        elif inputs[key][4] == blood_type[1]:
            blood_dict[blood_type[1]][key] = inputs[key]
        elif inputs[key][4] == blood_type[2]:
            blood_dict[blood_type[2]][key] = inputs[key]
        elif inputs[key][4] == blood_type[3]:
            blood_dict[blood_type[3]][key] = inputs[key]
        else:
            print("ERROR blood type")
    return blood_dict


def write_data(inputs=None, blood_dict=None, file_name='work.xlsx', sheet_name='sheet'):
    assert isinstance(inputs, defaultdict) or isinstance(inputs, dict)
    writer = pd.ExcelWriter(file_name)

    def gen_pd_data(inp):
        data_ = defaultdict(list)
        for i, col_ in enumerate(data_col):
            for k in inp:
                data_[col_].append(inp[k][i])
        return pd.DataFrame(data_)

    if blood_dict is not None:
        for key in blood_dict.keys():
            gen_pd_data(blood_dict[key]).to_excel(writer, sheet_name=key)

    if inputs is not None:
        gen_pd_data(inputs).to_excel(writer, sheet_name=sheet_name)
    writer.save()


def data_recovery(file, sheet='sheet'):
    pdata = pd.read_excel(file, sheet_name=sheet, converters={u'身份识别码': str})
    values_np = pdata.iloc[:].values
    values_np = np.delete(values_np, 0, axis=1)
    key_np = values_np[:, 0].tolist()
    values_np = values_np.tolist()
    return defaultdict(list, **dict(zip(key_np, values_np)))


def data_from_message(file, sheet='sheet'):
    print('reading message excel')
    day_list = []
    pdata = pd.read_excel(file, sheet_name=sheet, converters={u'识别码': str, u'身份识别码': str})
    time_list = pdata.iloc[:, 1].values
    time_list = list(set(time_list))
    for time in time_list:
        day_list.append(str(time).split(' ')[0])
    return list(set(day_list))


def select_data_from_message(timelist, xls_file_list, pos='yangzhou2'):
    result = None
    for file in xls_file_list[:-1]:
        xls_file = pd.ExcelFile(file)
        for sheet_name in xls_file.sheet_names:
            print('reading', sheet_name, '......')
            a = read_file(file, sheet_name)
            result = data_match(a, result)

    for time_index in range(len(timelist)):
        for file in [xls_file_list[-1]]:
            xls_file = pd.ExcelFile(file)
            for sheet_name in xls_file.sheet_names:
                print('reading', sheet_name, '......')
                a = read_file(file, sheet_name)
                if time_index == 0:
                    result = data_match(a, result, timelist[time_index])
                else:
                    result = data_match(a, result, timelist[time_index], timelist[time_index-1])
        # result_ = data_clip(copy.deepcopy(result), timelist[time_index])
        # may be wrong but fast and cheap memory
        result_ = data_clip(result, timelist[time_index])
        
        blood_dic = classify_data_by_blood(result_)
        # relative path will make ERROR in other dirs except main dir
        write_data(result_, blood_dic, './data/exp_data_xlsx/'+pos+'/'+timelist[time_index]+'.xlsx')


def gen_legal_sample(data_dic, time_now='now'):
    assert isinstance(data_dic, defaultdict) or isinstance(data_dic, dict)
    message_time = pd.Timestamp(pd.datetime.now()) if time_now == 'now' else pd.Timestamp(time_now)
    poplist = []
    age_reason = 0
    interval_reason = 0
    possible_reason = 0
    for key in data_dic.keys():
        age = (message_time - pd.Timestamp((data_dic[key][1]))).days // 365
        if age > 60 or data_dic[key][13] <= 180\
                or (data_dic[key][13] > 5 * data_dic[key][10] and data_dic[key][13] > 5000):
            poplist.append(key)
        if age > 60:
            age_reason = age_reason + 1
        if data_dic[key][13] < 180:
            interval_reason += 1
        # if data_dic[key][13] > 5 * data_dic[key][10] and data_dic[key][13] > 5000:
        #     possible_reason += 1
    for key in poplist:
        data_dic.pop(key)
    print('pop for age reason: ' + str(age_reason))
    print('pop for interval reason: ' + str(interval_reason))
    print('pop for possible reason: ' + str(possible_reason))
    return data_dic
