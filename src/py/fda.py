#	CS669 - Assignment 4 (Group-2) 
#	Last edit: 19/11/17
#	About: 
#		This program is a Bayes Classifier using GMM over reduced-dimensional representation of data usinf FDA.

import numpy as np
import math
import os
import sys
import random
import time
			
dimension=2									#	Dimension of data vectors.
classes=[]									#	Contains data of the class.
transformedClasses=[]						#	Contains training data with reduced dimensions.
testTransformedClasses=[]					#	Contains test data with reduced dimensions.
classesName=[]								#	Names of classes.
testClasses=[]								#	Contains test data.
totalData=[]								#	Concatenated training data.
covarianceMatrix=[]							#	Of the training data as a whole for Eigen Analysis.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of clusters during GMM.
clusters=[]									#	Stores data in clusters for all classes.
clusterMeans=[]								#	Stores cluster means for all classes.				
clusterPi=[]								#	Stores cluster mixing coefficients for GMM.
randSmall=[1e-310, 7.8e-309, 5.6e-308, 9.2e-312, 3.9e-310, 2.7e-307, 8.7e-309]		# Random small values for error handling.
randLarge=[3.4e310, 6.8e309, 7.6e308, 9.3e312, 9.9e310, 4.7e307, 8.8e309]			# Random large values for errro handling.

#	Calculates distance between two points in 'dimension' dimensional space.
def dist(x,y,dimensionPassed):
	distance=0
	for i in range(dimensionPassed):
		distance+=(x[i]-y[i])**2
	distance=math.sqrt(distance)
	return (distance)

#	Covariance term for given indices.
def covarianceTerm(i,j):
	Sum=0
	for k in range(len(totalData)):
		x=totalData[k]
		Sum+=(x[i]-Mean[i])*(x[j]-Mean[j])
	Sum/=len(totalData)
	return Sum

#	Returns the covaricance between dimension 'i' and 'j', of 'cluster' indexed cluster in class with index 'ind'.
def covarianceTermCluster(ind,cluster,i,j):
	Sum=0
	for k in range(len(clusters[ind][cluster])):
		x=clusters[ind][cluster][k]
		Sum+=(x[i]-clusterMeans[ind][cluster][i])*(x[j]-clusterMeans[ind][cluster][j])
	Sum/=len(clusters[ind][cluster])
	return Sum

#	Calculates the covariance matrix for all classes.
def calcCovarianceMat():
	global covarianceMatrix
	covarianceMatrix=[[0 for x in range(dimension)] for y in range(dimension)]
	for j in range(dimension):
		for k in range(dimension):
			if j<=k:
				covarianceMatrix[j][k]=covarianceTerm(j,k)
				covarianceMatrix[k][j]=covarianceMatrix[j][k]

#	Calculates covariance matrices of all clusters in class with index 'ind'.
def calcCovarianceMatClusters(ind,dimensionPassed):
	tempClusterCovarianceMatrices=[]
	for i in range(len(clusters[ind])):
		tempCovarianceMat=[[0 for k in range(dimensionPassed)] for j in range(dimensionPassed)]
		for j in range(dimensionPassed):
			for k in range(dimensionPassed):
				if j<=k:
					tempCovarianceMat[j][k]=covarianceTermCluster(ind,i,j,k)
					tempCovarianceMat[k][j]=tempCovarianceMat[j][k]
		tempClusterCovarianceMatrices.append(tempCovarianceMat)
	clusterCovarianceMatrices.append(tempClusterCovarianceMatrices)	

#	Return the likelihood of a sample point 'x', given Gaussian parameters 'uK' and 'sigmaK'.
def likelihood(x,uK,sigmaK,dimensionPassed):
	Denom=((((2*math.pi)**(dimensionPassed))*(math.fabs(np.linalg.det(sigmaK))))**0.5)
	if Denom==0:
		randomSmall=random.sample(range(0,6),1)
		Denom=randSmall[randomSmall[0]]
	elif math.isnan(Denom):
		randomLarge=random.sample(range(0,6),1)
		Denom=randLarge[randomLarge[0]]
	value=1.0/Denom
	temp=[0 for i in range(dimensionPassed)]
	mul=0
	sigmaInvK=np.asmatrix(sigmaK).I.A
	for i in range(dimensionPassed):
		for j in range(dimensionPassed):
			temp[i]+=(x[j]-uK[j])*sigmaInvK[j][i]
	for i in range(dimensionPassed):
		mul+=temp[i]*(x[i]-uK[i])
	if math.isnan(mul):
		randomSmall=random.sample(range(0,6),1)
		mul=randSmall[randomSmall[0]]
	if mul>500:
		mul=500
	elif mul<-500:
		mul=-500
	
	value*=math.exp(-0.5*mul)
	
	if value==float('inf'):
		randomLarge=random.sample(range(0,6),1)
		value=randLarge[randomLarge[0]]
	return value

#	Returns in the index of class with maximum likelihood of having the sample point 'x'.
def classifyLikelihood(x,K,dimensionPassed):
	val=[0 for i in range(len(clusterMeans))]
	for i in range(len(clusterMeans)):
		for k in range(K):
			val[i]+=clusterPi[i][k]*likelihood(x,clusterMeans[i][k],clusterCovarianceMatrices[i][k],dimensionPassed)
	return np.argmax(val)

#	K-means clustering for initiating GMM formation.
def kMeansClusteringandGMM(ind,K,dimensionPassed):

	tempClass=np.array(transformedClasses[ind])
	N=len(tempClass)

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0.0 for i in range(dimensionPassed)] for j in range(K)]
	tempClusterMean=np.array(tempClusterMean)
	randomKMeans=random.sample(range(0,N-1),K)
	for i in range(K):
		for j in range(dimensionPassed):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	# print tempClusterMean

	#	Dividing the data of this class to K clusters...
	tempClusters=[[] for i in range(K)]
	totDistance=0
	energy=np.inf
	for i in range(N):
		minDist=np.inf
		minDistInd=0
		for j in range(K):
			Dist=dist(tempClass[i],tempClusterMean[j],dimensionPassed)
			if Dist<minDist:
				minDist=Dist
				minDistInd=j
		tempClusters[minDistInd].append(tempClass[i])
		totDistance+=minDist

	#	Re-evaluating centres until the energy of changes becomes insignificant (convergence)...
	while energy>0.000001:
		tempClusterMean=[[0.0 for i in range(dimensionPassed)] for j in range(K)]
		tempClusterMean=np.array(tempClusterMean)
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimensionPassed):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimensionPassed):
				tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0
		for i in range(N):
			minDist=np.inf
			minDistInd=0
			for j in range(K):
				Dist=dist(tempClass[i],tempClusterMean[j],dimensionPassed)
				if Dist<minDist:
					minDist=Dist
					minDistInd=j
			tempClusters[minDistInd].append(tempClass[i])
			newTotDistance+=minDist
		energy=math.fabs(totDistance-newTotDistance)
		totDistance=newTotDistance

	clusters.append(tempClusters)
	clusterMeans.append(tempClusterMean)
	
	#	GMM.

	#	Calculating Covariance Matrices for all clusters...
	calcCovarianceMatClusters(ind,dimensionPassed)
	
	#	Calculating mixing coefficients for all clusters...
	tempClusterPi=[]
	for i in range(K):
		tempClusterPi.append(float(len(tempClusters[i]))/N)

	#	Gaussian Mixture Modelling...

	#	Using these initial calculated values for the EM algorithm.
	
	tempClusterCovarianceMatrices=clusterCovarianceMatrices[ind]
	energy=np.inf
	tempL=0
	iterations=1

	while energy>1 and iterations<100:
		
		#	Expectation step in the algorithm...
		tempGammaZ=[[0 for i in range (K)] for j in range (N)]
		tempLikelihoodTerms=[[0 for i in range(K)] for j in range(N)]
		tempDenom=[0 for i in range(N)]
		tempGammaSum=[0 for i in range(K)]
		newTempL=0

		#	Calculating responsibilty terms using previous values of parameters. 
		for n in range(N):
			for k in range(K):
				determinant=np.linalg.det(tempClusterCovarianceMatrices[k])
				while determinant==0:
					for i in range(dimensionPassed):
						tempClusterCovarianceMatrices[k][i][i]+=0.5
					determinant=np.linalg.det(tempClusterCovarianceMatrices[k])
				varLikelihood=likelihood(tempClass[n],tempClusterMean[k],tempClusterCovarianceMatrices[k],dimensionPassed)
				if varLikelihood==0:
					randomSmall=random.sample(range(0,6),1)
					varLikelihood=randSmall[randomSmall[0]]	
				tempLikelihoodTerms[n][k]=tempClusterPi[k]*varLikelihood
				if tempLikelihoodTerms[n][k]==0:
					randomSmall=random.sample(range(0,6),1)
					tempLikelihoodTerms[n][k]=randSmall[randomSmall[0]]	
				tempDenom[n]+=tempLikelihoodTerms[n][k]
			for k in range(K):
				tempGammaZ[n][k]=tempLikelihoodTerms[n][k]/tempDenom[n]
				tempGammaSum[k]+=tempGammaZ[n][k]

		#	Maximization step in the algorithm...
		#	Refining mean vectors.
		for k in range(K):
			for i in range(dimensionPassed):
				tempClusterMean[k][i]=0.0
				for n in range(N):
					tempClusterMean[k][i]+=tempGammaZ[n][k]*tempClass[n][i]
				tempClusterMean[k][i]/=tempGammaSum[k]

		#	Refining covariance matrices.
		for k in range(K):
			tempMatrix=[[0 for i in range(dimensionPassed)] for j in range(dimensionPassed)]
			for n in range(N):
				tempMatrix+=tempGammaZ[n][k]*np.outer((tempClass[n]-tempClusterMean[k]),(tempClass[n]-tempClusterMean[k]))
			tempMatrix/=tempGammaSum[k]
			determinant=np.linalg.det(tempMatrix)
			while determinant==0:
				for i in range(dimensionPassed):
					tempMatrix[i][i]+=0.5
				determinant=np.linalg.det(tempMatrix)
			if tempL==0:
				tempClusterCovarianceMatrices.append(tempMatrix)
			else:
				tempClusterCovarianceMatrices[k]=tempMatrix

		#	Refining mixing coefficients.
		for k in range(K):
			tempClusterPi[k]=tempGammaSum[k]/N

		for n in range(N):
			newTempL+=math.log(tempDenom[n])

		if tempL==0:
			tempL=newTempL
			iterations+=1
			continue
		else:
			energy=math.fabs(tempL-newTempL)
			tempL=newTempL
			iterations+=1

	del tempGammaSum,tempGammaZ,tempLikelihoodTerms,tempDenom
	clusterMeans[ind]=tempClusterMean
	clusterCovarianceMatrices[ind]=tempClusterCovarianceMatrices
	clusterPi.append(tempClusterPi)

#	Program starts here...
print ("\nThis program is a Bayes Classifier using GMM over reduced-dimensional representation of data using FDA.\n")

#	Parsing Input... 
choice= raw_input("\nDo you want to use your own directory for features training/test input and output or default (o/d): ")

direct=""
directO=""
directT=""
choiceIn=1
choiceInner='a'

if choice=='o':
	direct=raw_input("Enter the path (relative or complete) of the training feature data directory: ")
	directT=raw_input("Enter the path (relative or complete) of the test feature data directory: ")
	dimension=input("Enter the original number of dimensions in the data (for input format, refer README): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the classification: ")
else:
	choiceIn=input("Dataset (1/2): ")
	if choiceIn==1 or choiceIn==1:
		choiceInner=raw_input("Dataset 1(a/b): ")
		if choiceInner=='a' or choiceInner=='A':
			direct="../../../data/Input/Dataset 1/A/train"
			directT="../../../data/Input/Dataset 1/A/test"
			directO="../../data/Output/Dataset 1/A/test_results/"
		elif choiceInner=='b' or choiceInner=='B':
			direct="../../../data/Input/Dataset 1/B/train"
			directT="../../../data/Input/Dataset 1/B/test"
			directO="../../data/Output/Dataset 1/B/test_results/"
		else:
			print "Wrong input!. Exiting,"
			sys.exit()	
		dimension=2
	elif choiceIn==2 or choiceIn==2:
		direct="../../../data/Input/Dataset 2/train"
		directT="../../../data/Input/Dataset 2/test"
		directO="../../data/Output/Dataset 2/test_results/"
		dimension=64
	else:
		print "Wrong input!. Exiting,"
		sys.exit()

for filename in os.listdir(direct):
	file=open(os.path.join(direct,filename))
	tempClassData=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(num) for num in number_strings]
		tempClassData.append(numbers)
	classes.append(tempClassData)
	classesName.append(os.path.splitext(filename)[0])
	file.close()

for i in range(len(classesName)):
	for filename in os.listdir(directT):
		if os.path.splitext(filename)[0]==classesName[i]:
			file=open(os.path.join(directT,filename))
			tempClassData=[]
			for line in file:
				number_strings=line.split()
				numbers=[float(num) for num in number_strings]
				tempClassData.append(numbers)
			testClasses.append(tempClassData)
			file.close()


start_time=time.clock()

#	Clubbing all data to form a single set of data.
for ci in range(len(classes)):
	for x in range(len(classes[ci])):
		totalData.append(classes[ci][x])

#	Finding the mean vector of total data.
Mean=[0 for i in range(dimension)]
for x in range(len(totalData)):
	for d in range(dimension):
		Mean[d]+=totalData[x][d]
for d in range(dimension):
		Mean[d]/=len(totalData)

#	Finding the covariance matrix of total data.
calcCovarianceMat()

#	Finding EigenValues and EigenVectors for the covariance matrix.
eigenValues,eigenVectors=np.linalg.eigh(covarianceMatrix)

# Sort the (eigenValues, eigenVectors) tuples from high to low
eigPairs=[(np.abs(eigenValues[i]),eigenVectors[:,i]) for i in range(len(eigenValues))]
eigPairs.sort(key=lambda x: x[0], reverse=True)
eigenValueVectorTuples.append(eigPairs)

#	PCA for l(reduced dimensions) ranging from 1 to d-1
for l in range(64,65):
	transformedClasses=[]
	testTransformedClasses=[]

	for ci in range(len(classes)):
		newLDimensions=[]
		for k in range(l):
			newLDimensions.append(eigPairs[k][1])
		
		transformedData=[]
		for i in range(len(classes[ci])):
			newVector=[]
			for k in range(l):
				newVector.append(np.inner(np.array(classes[ci][i]),np.array(newLDimensions[k])))
			transformedData.append(newVector)
		
		testTransformedData=[]
		for i in range(len(testClasses[ci])):
			newVectorTest=[]
			for k in range(l):
				newVectorTest.append(np.inner(np.array(testClasses[ci][i]),np.array(newLDimensions[k])))
			testTransformedData.append(newVectorTest)
		
		transformedClasses.append(transformedData)
		testTransformedClasses.append(testTransformedData)
		del newLDimensions
		del transformedData
		del testTransformedData

	for k in range(4):
		clusters=[]
		clusterMeans=[]
		clusterCovarianceMatrices=[]
		clusterPi=[]

		for ci in range(len(classes)):
			kMeansClusteringandGMM(ci,2**k,l)
		confusionMatrix=[[0 for i in range(len(transformedClasses))] for j in range(len(transformedClasses))]
		for x in range(len(testTransformedClasses)):
			for y in range(len(testTransformedClasses[x])):
				ret=classifyLikelihood(testTransformedClasses[x][y],2**k,l)
				confusionMatrix[ret][x]+=1
		print confusionMatrix

		colors=['r','b','g']
		f=[]

		Sumtot=0
		for i in range(len(classes)):
			for j in range(len(classes)):
				Sumtot+=confusionMatrix[i][j]

		confusionMatClass=[]
		for i in range(len(classes)):
			tempConfusionMatClass=[[0 for j in range(2)] for p in range(2)]
			sumin=0
			tempConfusionMatClass[0][0]=confusionMatrix[i][i]
			sumin+=tempConfusionMatClass[0][0]
			Sum=0
			for j in range(len(classes)):
				Sum+=confusionMatrix[i][j]
			tempConfusionMatClass[0][1]=Sum-tempConfusionMatClass[0][0]
			sumin+=tempConfusionMatClass[0][1]
			Sum=0
			for j in range(len(classes)):
				Sum+=confusionMatrix[j][i]
			tempConfusionMatClass[1][0]=Sum-tempConfusionMatClass[0][0]
			sumin+=tempConfusionMatClass[1][0]
			tempConfusionMatClass[1][1]=Sumtot-sumin
			confusionMatClass.append(tempConfusionMatClass)
		
		print "Data testing complete. Writing results in files for future reference."
		filer=open(os.path.join(directO,"results_"+str(l)+"_"+str(k+1)+".txt"),"w")
		filev=open(os.path.join(directO,"values_"+str(k+1)+".txt"),"a")
		filet=open(os.path.join(directO,"times_"+str(k+1)+".txt"),"a")

		filer.write("The Confusion Matrix of all classes together is: \n")
		for i in range(len(classes)):
			for j in range(len(classes)):
				filer.write(str(confusionMatrix[i][j])+" ")
			filer.write("\n")

		filer.write("\nThe Confusion Matrices for different classes are: \n")
		for i in range(len(confusionMatClass)):
			filer.write("\nClass "+str(i+1)+": \n")
			for x in range(2):
				for y in range(2):
					filer.write(str(confusionMatClass[i][x][y])+" ")
				filer.write("\n")

		Accuracy=[]
		Precision=[]
		Recall=[]
		FMeasure=[]

		flagP,flagR,flagF=True,True,True
		filer.write("\nDifferent quantitative values are listed below.\n")
		for i in range(len(classes)):
			tp=confusionMatClass[i][0][0]
			fp=confusionMatClass[i][0][1]
			fn=confusionMatClass[i][1][0]
			tn=confusionMatClass[i][1][1]
			accuracy=float(tp+tn)/(tp+tn+fp+fn)
			if tp+fp:
				precision=float(tp)/(tp+fp)
			else:
				precision=0.0
				flagP=False
			if tp+fn:
				recall=float(tp)/(tp+fn)
			else:
				recall=0.0
				flagR=False
			if precision+recall:
				fMeasure=2*precision*recall/(precision+recall)
			else:
				fMeasure=0.0
				flagF=False

			filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
			if precision!=0.0:
				filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
			else:
				filer.write("Precision for class "+str(i+1)+" is -\n")
			if recall!=0.0:
				filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
			else:
				filer.write("Recall for class "+str(i+1)+" is -\n")
			if fMeasure!=0.0:
				filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
			else:
				filer.write("F-measure for class "+str(i+1)+" is -\n")
			Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

		avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
		for i in range (len(classes)):
			avgAccuracy+=Accuracy[i]
			avgPrecision+=Precision[i]
			avgRecall+=Recall[i]
			avgFMeasure+=FMeasure[i]
		avgAccuracy/=len(classes)
		avgPrecision/=len(classes)
		avgRecall/=len(classes)
		avgFMeasure/=len(classes)

		filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
		filev.write(str(avgAccuracy)+" ")
		filev.write(str(avgPrecision)+" ")
		filev.write(str(avgRecall)+" ")
		filev.write(str(avgFMeasure)+"\n")
		if flagP:
			filer.write("Average precision is "+str(avgPrecision)+"\n")
		else:
			filer.write("Average precision is -\n")
		if flagR:
			filer.write("Average recall is "+str(avgRecall)+"\n")
		else:
			filer.write("Average recall is -\n")
		if flagF:
			filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
		else:
			filer.write("Average F-Measure is -\n")
		filer.write("\n**End of results**")
		end_time=time.clock()
		diff=end_time-start_time
		filet.write(str(diff)+"\n")
		start_time=time.clock()
		filer.close()
		filev.close()
		filet.close()
		del confusionMatClass
		del confusionMatrix

#	End.