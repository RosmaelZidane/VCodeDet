import vulcodedetectmodel.inits.__inits__important as imp
import generatenodeedege as gg 
import torch     
import numpy as np    
from dgl import load_graphs, save_graphs
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
from dotenv import load_dotenv
import os
from flask import Flask, request, app, jsonify, url_for, render_template



app = Flask(__name__)




NUM_JOBS = 1  
JOB_ARRAY_NUMBER = 0 

# take the code from the user
def getjavacodetoanalyse(path_codea, text:str):
    try:
        with open(path_codea, 'w') as f:
            f.write(text)
        print("Java code provided to the analyser")
    except:
        print("The code is not provided")
    return



# load the save model checkpoint
checkpoint_path = f"{imp.external_dir()}/checkpoints/Vcodedet.ckpt"
model = gg.LitGNN.load_from_checkpoint(checkpoint_path)




@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])

def predict():# text: str
    """Provide a Java code for analysis.
    Important note: for successful graph generation the code/method/file must be between 'public class main{ <code> }'
    eg:
    public class main{
		private void complete(boolean dispose){
			if (dispose){
				try{
					saslServer.dispose();
				}
			}
		}
 	}"""
    java_code = request.form['java_source_code']
    path_codea = f"{imp.project_dir().parent.parent}/codeanalyse.java"
    getjavacodetoanalyse(path_codea, java_code)
    code_text = gg.readjavacode(pathjavacode = path_codea)[0]
    id = gg.readjavacode(pathjavacode= path_codea)[1]
    
    df = gg.createdf(code_text, id)
    df = df.iloc[::-1]
    splits = np.array_split(df, NUM_JOBS)
    if __name__ == "__main__":
        imp.dfmp(splits[JOB_ARRAY_NUMBER], gg.preprocess, ordr=False, workers=8)
    lines, graph_type, feat = gg.initialize_lines_and_features(gtype="pdg+raw", feat="all")
    codebert = gg.CodeBert()
    _id = id
    gg.cache_codebert_method_level(df, codebert, _id)
    g = gg.process_item(_id, df, codebert, lines, graph_type, feat)
    output = gg.modelpredict(model, g, id).transpose()
    
    output_html = output.to_html(classes='table table-striped', index=True)
    

    return render_template("home.html", prediction_text="Code Analysis: \n {}".format(output_html))
   


if __name__=="__main__":
    app.run(debug=True, port=8000)