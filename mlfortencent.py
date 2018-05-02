#coding=utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
import time
start_time = time.time()
user1=pd.read_csv('user1_feature.csv',index_col=[0,1])
user2=pd.read_csv('user2_feature.csv',index_col=[0,1])
labletrain=pd.DataFrame(np.array([1]*len(user1)+[2]*len(user2)),columns=['label'])
user1=user1[['age','gender','LBS','consumptionAbility','education']]
user2=user2[['age','gender','LBS','consumptionAbility','education']]
user1=user1.append(user2)

# user1=user1.drop(['uid'],axis=1)
# print(user1.head())

# def read_data():
#     user1=pd.read_csv('user1_feature.csv')
#     user2=pd.read_csv('user2_feature.csv')
#     user1=user1.drop['uid']
#     user=[]
#     user.append(user1)
#     user.append(user2)
#     return user

xgb_train=xgb.DMatrix(user1,label=labletrain)
# xgb_val=xgb.DMatrix()
# xgb_test = xgb.DMatrix()
params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'num_class':10, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 5000 # 迭代次数
watchlist = [(xgb_train, 'train')]
# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)


model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit)
# print("******************predict begin**********************")
# preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
cost_time = time.time()-start_time




