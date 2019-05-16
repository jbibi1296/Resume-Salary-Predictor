import resume_evaluator as rr
import pandas as pd
# jj = rr.check_all_worths(str(input("enter resume here")))
# models = [
# 'Linear Regression','Lasso','Ridge','Random Forest','Gradient Boost',
# 'Neural Net','Linear Regression Poly','Lasso Poly',
# 'Ridge Poly','Random Forest Poly','Gradient Boost Poly','Neural Net Poly',
# ]

# for i in models:
#     number = jj.T['worth'][i]
#     margin = jj.T['margin'][i]
#     print(f'{i} Model\n')
#     print(f'Your resume is worth ${round(number,2)} ± ${round(margin,2)}')
#     print('==============================\n\n')
def run_tester(text,model):
    lists = []
    mapper = {'Linear Regression':'lr',
     'Lasso':'ls',
     'Ridge':'rd',
     'Random Forest':'rf',
     'Gradient Boost':'gb',
     'Neural Network':'nn',
     'Linear Regression Poly':'lr_poly',
     'Lasso Poly':'ls_poly',
     'Ridge Poly':'rd_poly',
     'Random Forest Poly':'rf_poly',
     'Gradient Boost Poly':'gb_poly',
     'Neural Network Poly':'nn_poly'}
    jj = rr.check_your_worth(text,mapper[model])
    lists.append(jj)
    return lists