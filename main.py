from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

#For model
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

#trains the model when this file is run and makes it ready to use
df = pd.read_csv('https://drive.google.com/u/0/uc?id=1f2tq6Tcxjw-d99iH7N2WPb5gRzIcCjby&export=download')
df['class'] = df['class'].map(lambda x: 1 if x == 4 else 0)
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])
rfc = RandomForestClassifier(n_estimators=200, max_depth=140, max_features='sqrt')
rfc.fit(X,y)
y_pred=cross_val_predict(rfc, X, y, cv=10)
final_model = rfc.fit(X,y)

class BCD(Resource):

	#to accept post operations
	def post(self):

		#get the arguements from the post method
		bcd_put_Args = reqparse.RequestParser()
		bcd_put_Args.add_argument("Clump Thickness")
		bcd_put_Args.add_argument("Uniformity of cell size")
		bcd_put_Args.add_argument("Uniformity of cell shape")
		bcd_put_Args.add_argument("Marginal adhesion")
		bcd_put_Args.add_argument("Single epithelial cell size")
		bcd_put_Args.add_argument("Bare nuclei")
		bcd_put_Args.add_argument("Bland chromatin")
		bcd_put_Args.add_argument("Normal nuclei")
		bcd_put_Args.add_argument("Mitosis")
		args = bcd_put_Args.parse_args()

		#print the arguements just to make sure we are setup for predicting
		print(args)

		#predict a result for the new given arguements
		new_prediction=final_model.predict([[
			args["Clump Thickness"],
			args["Uniformity of cell size"],
			args["Uniformity of cell shape"],
			args["Marginal adhesion"],
			args["Single epithelial cell size"],
			args["Bare nuclei"],
			args["Bland chromatin"],
			args["Normal nuclei"],
			args["Mitosis"]
			]])

		#return the result in JSON format
		return {"result": int(new_prediction)}, 201

#should be below the class since we can't use it before defining the class
#adds a route to give a response for any given request
api.add_resource(BCD, "/api/v1/model/predict")

if __name__ == "__main__":
	app.run(debug=True)