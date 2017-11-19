#	CS669 - Assignment 4 (Group-2) 
#	Last edit: 20/11/17
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
classesName=[]								#	Names of classes.
testClasses=[]								#	Contains test data.
means=[]									#	Means of all classes
scatterWithin=[]							#	Within-class scatter matrix
scatterBetween=[]							#	Between-class scatter matrix
scatterMatrices=[]							#	Scatter Matrices of all classes
pairwiseEigen=[]							#	EigenValue-EigenVector pair for pairs of classes.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of clusters during GMM, temporarily.
clusters=[]									#	Stores data in clusters for all classes, temporarily.
clusterMeans=[]								#	Stores cluster means for all classes, temporarily.				
clusterPi=[]								#	Stores cluster mixing coefficients for GMM, temporarily.
bigClusters=[]								#	Stores data in clusters for all classes.
bigClusterMeans=[]							#	Stores cluster means for all classes.
bigClusterCovarianceMatrices=[]				#	Stores covariance matrices of clusters during GMM.
bigClusterPi=[]								#	Stores cluster mixing coefficients for GMM.
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
def scatterTerm(ind,i,j):
	Sum=0
	for k in range(len(classes[ind])):
		x=classes[ind][k]
		Sum+=(x[i]-means[ind][i])*(x[j]-means[ind][j])
	return Sum

#	Returns the covaricance between dimension 'i' and 'j', of 'cluster' indexed cluster in class with index 'ind'.
def covarianceTermCluster(classInd,cluster,i,j):
	Sum=0
	for k in range(len(clusters[classInd][cluster])):
		x=clusters[classInd][cluster][k]
		Sum+=(x[i]-clusterMeans[classInd][cluster][i])*(x[j]-clusterMeans[classInd][cluster][j])
	Sum/=len(clusters[classInd][cluster])
	return Sum

#	Calculates the covariance matrix for all classes.
def calcScatterMat(ind):
	tempScatterMatrix=[[0 for x in range(dimension)] for y in range(dimension)]
	for j in range(dimension):
		for k in range(dimension):
			if j<=k:
				tempScatterMatrix[j][k]=scatterTerm(ind,j,k)
				tempScatterMatrix[k][j]=tempScatterMatrix[j][k]
	scatterMatrices.append(np.array(tempScatterMatrix))

#	Calculates covariance matrices of all clusters in class with index 'ind'.
def calcCovarianceMatClusters(classInd,dimensionPassed):
	tempClusterCovarianceMatrices=[]
	for i in range(len(clusters[classInd])):
		tempCovarianceMat=[[0 for k in range(dimensionPassed)] for j in range(dimensionPassed)]
		for j in range(dimensionPassed):
			for k in range(dimensionPassed):
				if j<=k:
					tempCovarianceMat[j][k]=covarianceTermCluster(classInd,i,j,k)
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
def classifyLikelihood(ind,x,K,dimensionPassed):
	val=[0 for i in range(len(bigClusterMeans[ind]))]
	for i in range(len(bigClusterMeans[ind])):
		for k in range(K):
			val[i]+=bigClusterPi[ind][i][k]*likelihood(x,bigClusterMeans[ind][i][k],bigClusterCovarianceMatrices[ind][i][k],dimensionPassed)
	return np.argmax(val)

#	K-means clustering for initiating GMM formation.
def kMeansClusteringandGMM(ind,classInd,K,dimensionPassed):

	tempClass=[[0 for i in range(dimensionPassed)] for j in range(len(transformedClasses[ind][classInd]))] 
	for i in range(len(transformedClasses[ind][classInd])):
		for j in range(dimensionPassed):
			tempClass[i][j]=transformedClasses[ind][classInd][i][j]
	tempClass=np.array(tempClass)
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
	calcCovarianceMatClusters(classInd,dimensionPassed)
	
	#	Calculating mixing coefficients for all clusters...
	tempClusterPi=[]
	for i in range(K):
		tempClusterPi.append(float(len(tempClusters[i]))/N)

	#	Gaussian Mixture Modelling...

	#	Using these initial calculated values for the EM algorithm.
	
	tempClusterCovarianceMatrices=clusterCovarianceMatrices[classInd]
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
	clusterMeans[classInd]=tempClusterMean
	clusterCovarianceMatrices[classInd]=tempClusterCovarianceMatrices
	clusterPi.append(tempClusterPi)

def check(x,dimensionPassed,K,classInd1,classInd2,ind):
	global clusters,clusterMeans,clusterCovarianceMatrices,clusterPi

	newVectorTest=[]
	for k in range(dimensionPassed):
		newVectorTest.append(np.inner(np.array(x),np.array(pairwiseEigen[ind][k][1])))
	
	ret=classifyLikelihood(ind,newVectorTest,K,dimensionPassed)
	if ret==0:
		ret=classInd1
	else:
		ret=classInd2
	return ret

#	Program starts here...
print ("\nThis program is a Bayes Classifier using GMM over reduced-dimensional representation of data using FDA.")

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

#	Finding the mean vector of total data.
for ci in range(len(classes)):
	tempMean=[0 for i in range(dimension)]
	for x in range(len(classes[ci])):
		for d in range(dimension):
			tempMean[d]+=classes[ci][x][d]
	for d in range(dimension):
		tempMean[d]/=len(classes[ci])
	means.append(np.array(tempMean))

#	Finding the between-class scatter matrix for all pair of classes.
for i in range(len(classes)):
	for j in range(i+1,len(classes)):
		tempMat=np.outer((means[i]-means[j]),(means[i]-means[j]))
		for x in range(dimension):
			for y in range(dimension):
				tempMat[x][y]*=len(classes[i])
		scatterBetween.append(tempMat)

#	Finding the scatter matrices for all classes.
for i in range(len(classes)):
	calcScatterMat(i)

#	Finding the within-class scatter matrix for corresponding class-pairs.
for i in range(len(classes)):
	for j in range(i+1,len(classes)):
		scatterWithin.append(scatterMatrices[i]+scatterMatrices[j])

for ci in range(len(scatterBetween)):
	determinant=np.linalg.det(scatterWithin[ci])
	while determinant==0:
		for i in range(dimension):
			scatterWithin[ci][i][i]+=0.5
		determinant=np.linalg.det(scatterWithin[ci])
	determinant=np.linalg.det(scatterBetween[ci])
	while determinant==0:
		for i in range(dimension):
			scatterBetween[ci][i][i]+=0.5
		determinant=np.linalg.det(scatterBetween[ci])
	
	transformationMatrix=np.array((np.asmatrix(scatterWithin[ci]).I)*(np.asmatrix(scatterBetween[ci])))

	#	Finding EigenValues and EigenVectors for the transformation Matrix.
	eigenValues,eigenVectors=np.linalg.eigh(transformationMatrix)

	# Sort the (eigenValues, eigenVectors) tuples from high to low
	eigPairs=[(np.abs(eigenValues[i]),eigenVectors[:,i]) for i in range(len(eigenValues))]
	eigPairs.sort(key=lambda x: x[0], reverse=True)
	pairwiseEigen.append(eigPairs)
	classInd1=0
	classInd2=1
	if ci==0:
		classInd1=0
		classInd2=1
	elif ci==1:
		classInd1=0
		classInd2=2
	else:
		classInd1=1
		classInd2=2

	tempTransformedClasses=[]
	transformedData=[]
	for i in range(len(classes[classInd1])):
		newVector=[]
		for k in range(dimension):
			newVector.append(np.inner(np.array(classes[classInd1][i]),np.array(pairwiseEigen[ci][k][1])))
		transformedData.append(newVector)
	tempTransformedClasses.append(transformedData)
	transformedData=[]
	for i in range(len(classes[classInd2])):
		newVector=[]
		for k in range(dimension):
			newVector.append(np.inner(np.array(classes[classInd2][i]),np.array(pairwiseEigen[ci][k][1])))
		transformedData.append(newVector)
	tempTransformedClasses.append(transformedData)
	transformedClasses.append(tempTransformedClasses)


for l in range(64,65):
	for k in range(1,6):
		
		if l==1 and k==1:
			file=open(os.path.join(directO,"train_l_1_"+str(0)+"_"+str(1)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[0][i])):
					file.write(str(transformedClasses[0][i][j][0])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_1_"+str(0)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[1][i])):
					file.write(str(transformedClasses[1][i][j][0])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_1_"+str(1)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[2][i])):
					file.write(str(transformedClasses[2][i][j][0])+"\n")
			file.close()
		elif l==2 and k==1:
			file=open(os.path.join(directO,"train_l_2_"+str(0)+"_"+str(1)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[0][i])):
					file.write(str(transformedClasses[0][i][j][0])+" "+str(transformedClasses[0][i][j][1])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_2_"+str(0)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[1][i])):
					file.write(str(transformedClasses[1][i][j][0])+" "+str(transformedClasses[1][i][j][1])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_2_"+str(1)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[2][i])):
					file.write(str(transformedClasses[2][i][j][0])+" "+str(transformedClasses[2][i][j][1])+"\n")
			file.close()
		elif l==3 and k==1:
			file=open(os.path.join(directO,"train_l_3_"+str(0)+"_"+str(1)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[0][i])):
					file.write(str(transformedClasses[0][i][j][0])+" "+str(transformedClasses[0][i][j][1])+" "+str(transformedClasses[0][i][j][2])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_3_"+str(0)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[1][i])):
					file.write(str(transformedClasses[1][i][j][0])+" "+str(transformedClasses[1][i][j][1])+" "+str(transformedClasses[1][i][j][2])+"\n")
			file.close()
			file=open(os.path.join(directO,"train_l_3_"+str(1)+"_"+str(2)+".txt"),"w")
			for i in range(2):
				for j in range(len(transformedClasses[2][i])):
					file.write(str(transformedClasses[2][i][j][0])+" "+str(transformedClasses[2][i][j][1])+" "+str(transformedClasses[2][i][j][2])+"\n")
			file.close()

		bigClusters=[]
		bigClusterMeans=[]
		bigClusterCovarianceMatrices=[]
		bigClusterPi=[]

		for ind in range(3):
			clusters=[]
			clusterMeans=[]
			clusterCovarianceMatrices=[]
			clusterPi=[]
			for ci in range(len(transformedClasses[ind])):
				kMeansClusteringandGMM(ind,ci,k,l)
			bigClusters.append(clusters)
			bigClusterMeans.append(clusterMeans)
			bigClusterCovarianceMatrices.append(clusterCovarianceMatrices)
			bigClusterPi.append(clusterPi)


		confusionMatrix=[[0 for i in range(len(classes))] for j in range(len(classes))]
		for i in range(len(testClasses)):
			for j in range(len(testClasses[i])):
				x=testClasses[i][j]
				ret=check(x,l,k,0,1,0)
				if ret==0:
					ret=check(x,l,k,0,2,1)
				else:
					ret=check(x,l,k,1,2,2)
				print ret,i
				confusionMatrix[ret][i]+=1

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
		filer=open(os.path.join(directO,"results_"+str(l)+"_"+str(k)+".txt"),"w")
		filev=open(os.path.join(directO,"values_"+str(k)+".txt"),"a")
		filet=open(os.path.join(directO,"times_"+str(k)+".txt"),"a")

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