import csv
import pandas as pd
import util
import skfuzzy as fuzz
import numpy as np

class Read:
	def __init__(self, data_size, num_fuzz_var):
		self.data_size = data_size
		self.num_fuzz_var = num_fuzz_var

	def read_data(self):
		dataset = pd.read_csv('../xAPI-Edu-Data.csv')

		imp_features = dataset.drop(['gender', 'NationalITy', 'Semester', 'PlaceofBirth', 'GradeID', 'Topic', 'SectionID', 'Relation'], axis=1)
		stage_mapping = {"lowerlevel":0, "MiddleSchool":0.5, "HighSchool":1}
		survey_mapping = {"No":0, "Yes":1}
		satisfaction_mapping = {"Bad":0, "Good":1}
		absence_mapping = {"Under-7":0, "Above-7":1}
		class_mapping = {"L":0, "M":0.5, "H":1}
		numeric_features = imp_features
		numeric_features['StageID'] = imp_features['StageID'].map(stage_mapping)
		numeric_features['ParentAnsweringSurvey'] = imp_features['ParentAnsweringSurvey'].map(survey_mapping)
		numeric_features['ParentschoolSatisfaction'] = imp_features['ParentschoolSatisfaction'].map(satisfaction_mapping)
		numeric_features['StudentAbsenceDays'] = imp_features['StudentAbsenceDays'].map(absence_mapping)
		numeric_features['Class'] = imp_features['Class'].map(class_mapping)

		data_np_array = numeric_features.values
		print("Data size = ",data_np_array.shape)
		self.X1 = data_np_array[:,1:5]
		self.X2 = np.concatenate((data_np_array[:,0:1],data_np_array[:,5:8]),axis=1)
		self.Y = data_np_array[:,data_np_array.shape[1]-1:data_np_array.shape[1]]


	def prepare_data(self,mf, rng):
		x = np.zeros((self.data_size,self.num_fuzz_var*5))

		for i in range(self.data_size):
			for j in range(self.num_fuzz_var):
				x[i,j*5]=fuzz.interp_membership(rng, mf['vlo'], self.X1[i,j])
				x[i,j*5+1]=fuzz.interp_membership(rng, mf['lo'], self.X1[i,j])
				x[i,j*5+2]=fuzz.interp_membership(rng, mf['md'], self.X1[i,j])
				x[i,j*5+3]=fuzz.interp_membership(rng, mf['hi'], self.X1[i,j])
				x[i,j*5+4]=fuzz.interp_membership(rng, mf['vhi'], self.X1[i,j])

		self.X=np.concatenate((x,self.X2),axis=1)

