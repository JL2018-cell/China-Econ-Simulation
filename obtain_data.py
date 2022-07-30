# Read data to provide parameters for model.
#Read data intil 2015.
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

def obtain_data(data_path, time_low_bound = None, time_up_bound = None):
    # FInd path of data
    print("In obtain_data")

    root, directory, files = list(os.walk(data_path))[0]
    # Define names of agents.
    INDUSTRIES_CHIN = ["农林牧渔业", "电力、热力、燃气及水生产和供应业", "金融业", "信息传输、软件和信息技术服务业",  "采矿业", "住宿和餐饮业", "制造业", "建筑业", "交通运输、仓储和邮政业", "批发和零售业", "教育业"]
    INDUSTRIES = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'Manufacturing', 'Construction', 'Transport', 'Retail', 'Education']
    INDUSTRIES_DICT = dict(zip(INDUSTRIES_CHIN, INDUSTRIES))
    PROVINCES = {'北京市': "BeiJing", '天津市': "TianJin", '河北省': "HeBei", '山西省': "ShanXi1", '内蒙古自治区': "NeiMengGu", '辽宁省': "LiaoNing", '吉林省': "JiLin", '黑龙江省': "HeiLongJiang", '上海市': "ShangHai", '江苏省': "JiangSu", '浙江省': "ZheJiang", '安徽省': "AnHui", '福建省': "FuJian", '江西省': "JiangXi", '山东省': "ShanDong", '河南省': "HeNan", '湖北省': "HuBei", '湖南省': "HuNan", '广东省': "GuangDong", '广西壮族自治区': "GuangXi", '海南省': "HaiNan", '重庆市': "ChongQin", '四川省': "SiChuan", '贵州省': "GuiZhou", '云南省': "YunNan", '西藏自治区': "XiZang", '陕西省': "ShanXi", '甘肃省': "GanSu", '青海省': "QingHai", '宁夏回族自治区': "NingXia", '新疆维吾尔自治区': "XinJiang", '数据来源：国家统计局': "GuoJia"}
    PROVINCES_in_simul = ["GuangDong", "HeBei", "XinJiang", "AnHui", "ZheJiang", "SiChuan", "FuJian", "HuBei", "JiangSu", "ShanDong", "HuNan", "HeNan", "ShanXi"]
    unused_provinces = ['山西省', '内蒙古自治区', '辽宁省']
    # Prepare container of parameters
    contribution = {"GDP": {}, "CO2": {}, "resource_points": {}}
    industry_init_dstr = {}
    # Change of industry distribution over time.
    industry_dstr = {}
    # Change of GDP over time.
    GDP_series = {}
    # Change of CO2 over time.
    CO2_series = {}

    # Find relationship between variable of interest and industries labour distribution 
    def regression(X, y):
        snd_min = np.sort(y)[1]
        if np.min(y) == 0:
            y = np.clip(y, snd_min / 10000, None)
        else:
            y = np.clip(y, snd_min / 10000, None)
        model = LinearRegression().fit(X, np.apply_along_axis(np.log, 0, y))
        return model.intercept_, model.coef_

    # Make sure permutation of arr (array) is same as target_arr.
    def align(arr, target_arr):
        assert len(arr) == len(target_arr)
        indices = []
        for elm in target_arr:
            for el in arr:
                if elm in el:
                    indices.append(el)
        return indices

    # Find industry labour distribution of each province.
    labour_files = [file for file in files if "labour" in file.lower()]
    for labour_file in labour_files:
        data = pd.read_excel(root + r"/" + labour_file, header = 3, index_col = 0)
        if time_up_bound is not None and time_low_bound is not None:
            data = data[[col for col in data.columns if col < time_up_bound and (col > time_low_bound or col == time_low_bound)]]
        elif time_up_bound is None and time_low_bound is not None:
            data = data[[col for col in data.columns if col > time_low_bound or col == time_low_bound]]
        elif time_up_bound is not None and time_low_bound is None:
            data = data[[col for col in data.columns if col < time_up_bound]]
        [category, province] = labour_file[:labour_file.find(".")].split("_")
        target_indices = [index for index in data.index if any(industry in index for industry in INDUSTRIES_CHIN)]
        # Ensure arrangement of indices is the same as global variable INDUSTRIES_CHIN.
        target_indices = align(target_indices, INDUSTRIES_CHIN)
        state = data.loc[target_indices].fillna(method ='backfill', axis = 1)
        industry_init_dstr[province] = dict(zip(INDUSTRIES, state.loc[:, sorted(state.columns)[-1]].to_numpy()))
        industry_dstr[province] = state.dropna(axis = 1, how = 'all').T

    # Caculate remianing data. e.g. pollutants, resource points, GDP.
    for file in files:
        data = pd.read_excel(root + r"/" + file, header = 3, index_col = 0)
        if time_up_bound is not None and time_low_bound is not None:
            data = data[[col for col in data.columns if col < time_up_bound and (col > time_low_bound or col == time_low_bound)]]
        elif time_up_bound is None and time_low_bound is not None:
            data = data[[col for col in data.columns if col > time_low_bound or col == time_low_bound]]
        elif time_up_bound is not None and time_low_bound is None:
            data = data[[col for col in data.columns if col < time_up_bound]]
        [category, province] = file[:file.find(".")].split("_")
        if category.lower() == "sewage":
            y = data.fillna(method = 'bfill', axis = 1).fillna(method = 'ffill', axis = 1).fillna(0).sum()
            try:
                CO2_series[province] += y
            except KeyError:
                CO2_series[province] = y
            X = industry_dstr[province]
            X = X.join(y.rename("Sewage"), how = "inner")
            indices = [[col for col in X.columns if industry in col][0] for industry in INDUSTRIES_CHIN]
            indices = align(indices, INDUSTRIES_CHIN)
            X = X[indices + ["Sewage"]]
            bias, coeff = regression(X.to_numpy()[:, 0 : X.shape[1] - 1], X.to_numpy()[:, -1])
            # contribution["sewage"][province] = dict(zip(INDUSTRIES, coeff))
            contribution["CO2"][province] = {}
            industries_contrib = {**dict(zip(INDUSTRIES, coeff)), **{"bias": bias}}
            for k, v in industries_contrib.items():
                try:
                    contribution["CO2"][province][k] += v
                except KeyError:
                    contribution["CO2"][province][k] = v

        elif category.lower() == "air":
            y = data.fillna(method = 'bfill', axis = 1).fillna(method = 'ffill', axis = 1).fillna(0).sum()
            #y = data.fillna(0).sum()
            try:
                CO2_series[province] += y
            except KeyError:
                CO2_series[province] = y
            X = industry_dstr[province]
            X = X.join(y.rename("Air"), how = "inner")
            indices = [[col for col in X.columns if industry in col][0] for industry in INDUSTRIES_CHIN]
            indices = align(indices, INDUSTRIES_CHIN)
            X = X[indices + ["Air"]]
            #X = X[[[col for col in X.columns if industry in col][0] for industry in INDUSTRIES_CHIN] + ["Air"]]
            bias, coeff = regression(X.to_numpy()[:, 0 : X.shape[1] - 1], X.to_numpy()[:, -1])
            # contribution["air"][province] = dict(zip(INDUSTRIES, coeff))
            contribution["CO2"][province] = {}
            industries_contrib = {**dict(zip(INDUSTRIES, coeff)), **{"bias": bias}}
            for k, v in industries_contrib.items():
                try:
                    contribution["CO2"][province][k] += v
                except KeyError:
                    contribution["CO2"][province][k] = v

        elif category.lower() == "gdp":
            y = data.loc[[index for index in data.index if "地区生产总值" in index][0]]
            GDP_series[province] = y
            X = industry_dstr[province]
            X = X.join(y.rename("GDP"), how = "inner")
            X = X[[[col for col in X.columns if industry in col][0] for industry in INDUSTRIES_CHIN] + ["GDP"]]
            bias, coeff = regression(X.to_numpy()[:, 0 : X.shape[1] - 1], X.to_numpy()[:, -1])
            contribution["GDP"][province] = {**dict(zip(INDUSTRIES, coeff)), **{"bias": bias}}

        elif category.lower() == "labour":
            pass

        # Tax_income, deleted "budgeted income" at the first row of raw data.
        # Used to calculate resource points.
        else:
            data = data.fillna(method = "backfill", axis = 1)
            data_in_simul = data.loc[[index for index in data.index if PROVINCES[index] in PROVINCES_in_simul]]
            for i, index in enumerate(data_in_simul.index):
                X = industry_dstr[PROVINCES[index]]
                y = data_in_simul.loc[index]
                X = X.join(y.rename("resource_points"), how = "inner")
                bias, coeff = regression(X.to_numpy()[:, 0 : X.shape[1] - 1], X.to_numpy()[:, -1])
                contribution["resource_points"][PROVINCES[index]] = {**dict(zip(INDUSTRIES, coeff)), **{"bias": bias}}
    return (CO2_series, GDP_series, industry_dstr, industry_init_dstr, contribution) 

# Return labour distribution of different industries over time.
def industry_dstr_over_time(data_path, time_low_bound = None, time_up_bound = None):

    # Find path of data
    root, directory, files = list(os.walk(data_path))[0]
    # Define names of agents.
    INDUSTRIES_CHIN = ["农林牧渔业", "电力、热力、燃气及水生产和供应业", "金融业", "信息传输、软件和信息技术服务业",  "采矿业", "住宿和餐饮业", "制造业", "建筑业", "交通运输、仓储和邮政业", "批发和零售业", "教育业"]
    INDUSTRIES = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'Manufacturing', 'Construction', 'Transport', 'Retail', 'Education']
    INDUSTRIES_DICT = dict(zip(INDUSTRIES_CHIN, INDUSTRIES))
    PROVINCES = {'北京市': "BeiJing", '天津市': "TianJin", '河北省': "HeBei", '山西省': "ShanXi1", '内蒙古自治区': "NeiMengGu", '辽宁省': "LiaoNing", '吉林省': "JiLin", '黑龙江省': "HeiLongJiang", '上海市': "ShangHai", '江苏省': "JiangSu", '浙江省': "ZheJiang", '安徽省': "AnHui", '福建省': "FuJian", '江西省': "JiangXi", '山东省': "ShanDong", '河南省': "HeNan", '湖北省': "HuBei", '湖南省': "HuNan", '广东省': "GuangDong", '广西壮族自治区': "GuangXi", '海南省': "HaiNan", '重庆市': "ChongQin", '四川省': "SiChuan", '贵州省': "GuiZhou", '云南省': "YunNan", '西藏自治区': "XiZang", '陕西省': "ShanXi", '甘肃省': "GanSu", '青海省': "QingHai", '宁夏回族自治区': "NingXia", '新疆维吾尔自治区': "XinJiang", '数据来源：国家统计局': "GuoJia"}
    PROVINCES_in_simul = ["GuangDong", "HeBei", "XinJiang", "AnHui", "ZheJiang", "SiChuan", "FuJian", "HuBei", "JiangSu", "ShanDong", "HuNan", "HeNan", "ShanXi"]
    unused_provinces = ['山西省', '内蒙古自治区', '辽宁省']
    # Change of industry distribution over time.
    industry_dstr = {}

    # Make sure permutation of arr (array) is same as target_arr.
    def align(arr, target_arr):
        assert len(arr) == len(target_arr)
        indices = []
        for elm in target_arr:
            for el in arr:
                if elm in el:
                    indices.append(el)
        return indices

    # Find industry labour distribution of each province.
    labour_files = [file for file in files if "labour" in file.lower()]
    for labour_file in labour_files:
        data = pd.read_excel(root + r"/" + labour_file, header = 3, index_col = 0)
        if time_up_bound is not None and time_low_bound is not None:
            data = data[[col for col in data.columns if col < time_up_bound and (col > time_low_bound or col == time_low_bound)]]
        elif time_up_bound is None and time_low_bound is not None:
            data = data[[col for col in data.columns if col > time_low_bound or col == time_low_bound]]
        elif time_up_bound is not None and time_low_bound is None:
            data = data[[col for col in data.columns if col < time_up_bound]]
        print(labour_file, data.shape)
        [category, province] = labour_file[:labour_file.find(".")].split("_")
        target_indices = [index for index in data.index if any(industry in index for industry in INDUSTRIES_CHIN)]
        # Ensure arrangement of indices is the same as global variable INDUSTRIES_CHIN.
        target_indices = align(target_indices, INDUSTRIES_CHIN)
        state = data.loc[target_indices].fillna(method ='backfill', axis = 1)
        industry_dstr[province] = state.dropna(axis = 1, how = 'all').T

    return industry_dstr
