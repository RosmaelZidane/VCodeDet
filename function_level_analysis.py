import pandas as pd   

dpath = "/home/rz.lekeufack/Rosmael/Explainability_ex/cdata/VCodeDet/vulcodedetectmodel/storage/outputs/Analysis_by_func.csv"
df = pd.read_csv(dpath)
# df.reset_index(drop=True, inplace=True)
# print(df.head(5))
df = df[df['Func id'] != 178103]
df = df[df['Func id'] != 59010]
df.reset_index(drop=True, inplace=True)

Preduct_func = []
sa = len(df)
for i in range(len(df)): # len(df)
    stat_pred = df['preduc_by_func'][i]
    try:
        listL = list(map(int, stat_pred.strip('[]').split()))
    except:
        print(df['Func id'][i], stat_pred)
        
    if 1 in listL:
        Preduct_func.append(1)
    else:
        Preduct_func.append(0)

# print(df)
# print(len(Preduct_func))      

df['Func-level-pred'] = Preduct_func
df["Ground truth"] = df['func_label'] 
df['Stat-prediction'] = df['preduc_by_func']

keep_col = ['Func id', "Ground truth", 'Stat-prediction', 'Func-level-pred']


df = df[keep_col]

from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

y_true = df['Ground truth']
y_pred = df['Func-level-pred'] 

print(f"\nFunction level prediction\n \nTotal functions tested: {len(df)}.\nNumber of Vulnerable function: {len(df[df['Ground truth'] == 1])}")
precision = precision_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")

funcmetric = pd.DataFrame({"Accuracy": [accuracy],  "Precision": [precision],
                           "F1 Score": [f1], "Recall": [recall]})

funcmetric.to_csv("/home/rz.lekeufack/Rosmael/Explainability_ex/cdata/VCodeDet/vulcodedetectmodel/storage/outputs/func-level-metrics.csv", index = False)