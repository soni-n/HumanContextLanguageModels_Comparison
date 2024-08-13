import sys
import sqlalchemy
from sqlalchemy import create_engine
import math
import numpy as np
from os import listdir
import pandas as pd
from scipy.stats import t, ttest_rel
import argparse
from sqlalchemy.engine.url import URL

def open_csv(table):
     df = pd.read_csv(table)
     return df

def open_db(db, table, conn=None):
    if conn is None:
        myDB = URL(drivername='mysql', host='127.0.0.1', database=db,
                query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
        engine = create_engine(myDB, encoding='utf8')
        conn = engine.connect()
    select = f"select * from {table}"
    select = conn.execute(select)
    df = pd.DataFrame(select.fetchall())
    df.columns = select.keys()

    return df, conn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Set params")
    parser.add_argument('--db', type=str, help='name of database to connect to')
    parser.add_argument('--t1', type=str, help='name of table for model 1 preds')
    parser.add_argument('--t2', type=str, help='name of table for model 2 preds')
    parser.add_argument('--is_csv', default=False, type=bool, help='input csv file')
    parser.add_argument('--sort_by', type=str, help='field to sort record pairs by (i.e. Id or label)')
    parser.add_argument('--pred_field', type=str, help='field with the predictions')
    parser.add_argument('--label_field', type=str, help='field with original values')
    args = parser.parse_args()
 
    
    # GOAL: ONLY DATA LOAD IS IF STATEMENT
    if args.is_csv:
        m1,m2 = open_csv(args.t1),open_csv(args.t2)
    else:
        m1, conn = open_db(args.db, args.t1)
        m2, _ = open_db(args.db, args.t2, conn)
      
    m1_sorted = m1.sort_values(by=[args.sort_by])
    m2_sorted = m2.sort_values(by=[args.sort_by])
      
    print ('First 10 data values...')
    print (m1_sorted.head(10))
    print (m2_sorted.head(10))
        
    n = len(m1)
        
    AE_m1 = np.abs(m1_sorted[args.label_field] - m1_sorted[args.pred_field])
    AE_m2 = np.abs(m2_sorted[args.label_field] - m2_sorted[args.pred_field])
    
    y_diff = AE_m1 - AE_m2
    y_diff_mean, y_diff_std = np.mean(y_diff), np.std(y_diff)
    y_diff_t = y_diff_mean / (y_diff_std / np.sqrt(n))
    y_diff_p = t.sf(np.abs(y_diff_t), n-1)
    print('Ttest from scratch: (%.4f, %.4f)'%(y_diff_t, y_diff_p))
