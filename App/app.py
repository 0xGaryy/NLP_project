import streamlit as st 
import altair as alt
import plotly.express as px 

import pandas as pd 
import numpy as np 
from datetime import datetime


import joblib 

from google.cloud import storage
from tempfile import TemporaryFile
from csv import writer
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()



storage_client = storage.Client()
bucket_name=os.environ.get("bucket_name")
model_bucket=os.environ.get("model_file")

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_bucket)
with TemporaryFile() as temp_file:
    blob.download_to_file(temp_file)
    temp_file.seek(0)
    pipe_lr=joblib.load(temp_file)




def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = { "anger":"😠",
					    "disgust":"🤮",
					    "fear":"😨😱",
						"happy":"🤗",
						"joy":"😂",
						"neutral":"😐",
						"sad":"😔", 
						"sadness":"😔",
						"shame":"😳",
						"surprise":"😮"
					}


def main():
	st.title("Emotion Classifier")
	menu = ["Home","Monitor"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.beta_columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)

  
			with open('monitoring.csv', 'a') as f_object:
				writer_object = writer(f_object)
				writer_object.writerow([raw_text,prediction,np.max(probability),datetime.now().strftime("%d/%m/%Y %H:%M:%S")])
				f_object.close()
			

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		st.subheader("Monitor App")

		with st.beta_expander("Page Metrics"):
			
			df = pd.read_csv("./monitoring.csv")  

			st.write(df)  
	
	else:
		st.subheader("About")


if __name__ == '__main__':
	main()