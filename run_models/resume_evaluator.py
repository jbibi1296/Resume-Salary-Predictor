import clean as cl
import pickle 
import pandas as pd
import numpy as np
from keras.models import load_model

lr = pickle.load( open( "./models/original/Linear_model.pkl", "rb" ) )
ls = pickle.load( open( "./models/original/Lasso_model.pkl", "rb" ) )
rd = pickle.load( open( "./models/original/Ridge_model.pkl", "rb" ) )
rf = pickle.load( open( "./models/original/Random_Forest_model.pkl", "rb" ) )
gb = pickle.load( open( "./models/original/Gradient_Boost_model.pkl", "rb" ) )
nn = load_model("./models/original/Neural_Net.h5")

lr_poly = pickle.load( open( "./models/poly/Linear_model.pkl", "rb" ) )
ls_poly = pickle.load( open( "./models/poly/Lasso_model.pkl", "rb" ) )
rd_poly = pickle.load( open( "./models/poly/Ridge_model.pkl", "rb" ) )
rf_poly = pickle.load( open( "./models/poly/Random_Forest_model.pkl", "rb" ) )
gb_poly = pickle.load( open( "./models/poly/Gradient_Boost_model.pkl", "rb" ) )
nn_poly = load_model("./models/poly/Neural_Net.h5")

lr_margin = pickle.load( open( "./models/original/margins/lr_margin.pkl", "rb" ) )
ls_margin = pickle.load( open( "./models/original/margins/ls_margin.pkl", "rb" ) )
rd_margin = pickle.load( open( "./models/original/margins/rd_margin.pkl", "rb" ) )
rf_margin = pickle.load( open( "./models/original/margins/rf_margin.pkl", "rb" ) )
gb_margin = pickle.load( open( "./models/original/margins/gb_margin.pkl", "rb" ) )
nn_margin = pickle.load( open( "./models/original/margins/nn_margin.pkl", "rb" ) )

lr_poly_margin = pickle.load( open( "./models/poly/margins/lr_margin.pkl", "rb" ) )
ls_poly_margin = pickle.load( open( "./models/poly/margins/ls_margin.pkl", "rb" ) )
rd_poly_margin = pickle.load( open( "./models/poly/margins/rd_margin.pkl", "rb" ) )
rf_poly_margin = pickle.load( open( "./models/poly/margins/rf_margin.pkl", "rb" ) )
gb_poly_margin = pickle.load( open( "./models/poly/margins/gb_margin.pkl", "rb" ) )
nn_poly_margin = pickle.load( open( "./models/poly/margins/nn_margin.pkl", "rb" ) )


stop_words = pickle.load( open( "./models/word cleaning/custom_stop_words.pkl", "rb" ) )
body = pickle.load( open( "./models/word cleaning/body.pkl", "rb" ) )
title = pickle.load( open( "./models/word cleaning/title.pkl", "rb" ) )
location = pickle.load( open( "./models/word cleaning/location.pkl", "rb" ) )
poly = pickle.load( open("./models/poly/poly_features.pkl", "rb"))
pca = pickle.load( open("./models/poly/pca.pkl", "rb"))

body_non_poly =pickle.load(open( "./models/word cleaning/non_poly_body.pkl", "rb" ) )
title_non_poly =pickle.load(open( "./models/word cleaning/non_poly_title.pkl", "rb" ) )
location_non_poly =pickle.load(open( "./models/word cleaning/non_poly_location.pkl", "rb"))

def prepare_text_poly(text):
    
    text = cl.token_stop_lemm(text,stop_words)
    text = pd.DataFrame([{'body':text}])['body']
    text = body.transform(text)
    text = pd.DataFrame(data = text.todense(),columns = body.get_feature_names())

    vectors_titles = title.transform(text)
    title_df = pd.DataFrame(data = vectors_titles.toarray(), columns = title.get_feature_names())
    text = pd.merge(text,title_df,how = 'outer', left_index=True,right_index=True)

    vectors_location = location.transform(text)
    location_df = pd.DataFrame(data = vectors_location.toarray(), columns = location.get_feature_names())
    text = pd.merge(text,location_df,how = 'outer', left_index=True,right_index=True)
    text = text.iloc[[0]]
    text = poly.transform(text)
    text = pca.transform(text)
    return pd.DataFrame(text)

def prepare_text(text):
    
    text = cl.token_stop_lemm(text,stop_words)
    text = pd.DataFrame([{'body':text}])['body']
    text = body_non_poly.transform(text)
    text = pd.DataFrame(data = text.todense(),columns = body_non_poly.get_feature_names())

    vectors_titles = title_non_poly.transform(text)
    title_df = pd.DataFrame(data = vectors_titles.toarray(), columns = title_non_poly.get_feature_names())
    text = pd.merge(text,title_df,how = 'outer', left_index=True,right_index=True)

    vectors_location = location_non_poly.transform(text)
    location_df = pd.DataFrame(data = vectors_location.toarray(), columns = location_non_poly.get_feature_names())
    text = pd.merge(text,location_df,how = 'outer', left_index=True,right_index=True)
    text = text.iloc[[0]]
    return pd.DataFrame(text)

def check_your_worth(paragraph,model):
    text = paragraph
    if model[-1] == 'y':
        text = prepare_text_poly(text)
    else:
        text = prepare_text(text)
    
    if model[:2] == 'nn':
        worth = np.exp(eval(model).predict(text)[0][0])
    else:
        worth = np.exp(eval(model).predict(text)[0])
    
    worth_num = round(worth,0)
#     print(f'You are worth between ${round(worth-gb_margin,2)} and ${round(worth+gb_margin,2)}')
    margin = eval(f'{model}_margin')
    margin = round(margin,0)
    lists = []
    
    lower_margin = round(worth - margin,0)
    upper_margin = round(worth + margin,0)
    
    worth = str(worth_num)[:-5] + ','+ str(worth_num)[-5:]
    lower_margin = str(lower_margin)[:-5] + ','+ str(lower_margin)[-5:]
    upper_margin = str(upper_margin)[:-5] + ','+ str(upper_margin)[-5:]
    lists.append(worth)
    lists.append(lower_margin)
    lists.append(upper_margin)
    lists.append(worth_num)
    return lists

def check_all_worths(text):
    
    column_remap = {
    0:'Linear Regression',
    1:'Lasso',
    2:'Ridge',
    3:'Random Forest',
    4:'Gradient Boost',
    5:'Neural Net',
    6:'Linear Regression Poly',
    7:'Lasso Poly',
    8:'Ridge Poly',
    9:'Random Forest Poly',
    10:'Gradient Boost Poly',
    11:'Neural Net Poly',
    }
    pre_df_list = []
    for i in ['lr','ls','rd','rf','gb','nn','lr_poly','ls_poly',
              'rd_poly','rf_poly','gb_poly','nn_poly']:
        dic = {}
        dic['worth'] = check_your_worth(text,i)[0]
        dic['margin'] = eval(f'{i}_margin')
        pre_df_list.append(dic)
    return pd.DataFrame(pre_df_list).T.rename(columns =column_remap)
