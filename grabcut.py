import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import sklearn
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import igraph
import math

import directories
from visualization import *

ImagesAndBBoxes = directories.loadImagesAndBBoxes()
directories.ensure_dir(directories.output)

def minCuts(vertices, edges, capacities):
	assert len(edges) == len(capacities)
	assert min(capacities) >= 0
	
	capacities = [int(cap * 10000) for cap in capacities]
	
	#Now I need to make the graph
	g = igraph.Graph(directed=False)

	g.add_vertices(map(str, vertices))
	g.add_vertices(["source", "sink"])
	
	g.add_edges([map(str, edge) for edge in edges])

	#igraph.plot(g.as_undirected(), layout="fr", vertex_label=None, edge_width=[2 * cap for cap in capacityList])

	return g.as_directed().all_st_mincuts("source", "sink", capacity=capacities * 2)

def normalizeList(l):
	l = np.asarray([l])
	l = sklearn.preprocessing.normalize(l, norm='l1', axis=1, copy=True)
	l = list(l[0])
	return [x * 100 for x in l]

def normalizeArr(arr):
	return np.reshape(np.asarray(normalizeList(arr.flatten())), arr.shape)
"""
def normalizeArr(arr):
	arr -= np.min(arr)
	arr /= np.max(arr)
"""
def initBinaryEdges(img):
	edges = np.zeros(img.shape[:2], dtype="float")
	
	for apertureSize in [3, 5, 7]:	
		for minEdgeStrength in range(0, 500, 50):
			edges += cv2.Canny(image=img, threshold1=2*minEdgeStrength, threshold2=minEdgeStrength, L2gradient=True, apertureSize=apertureSize)
		
		#visualize(edges)
		
	edges = edges ** 4
	normalizeArr(edges)

	print("Creating binary edges...")
	verticalEdges = [[(y, x), (y + 1, x)] for x in range(img.shape[1]) for y in range(img.shape[0] - 1)]
	horisontalEdges = [[(y, x), (y, x + 1)] for x in range(img.shape[1] - 1) for y in range(img.shape[0])]
	de = [[(y, x), (y + 1, x + 1)] for x in range(img.shape[1] - 1) for y in range(img.shape[0] - 1)]
	de2 = [[(y, x), (y + 1, x - 1)] for x in range(1, img.shape[1]) for y in range(img.shape[0] - 1)]
	binaryEdges = horisontalEdges + verticalEdges + de + de2

	binaryCapacities = []
	for pts in binaryEdges:
		#cap = binaryCostFunction(img[pts[0]], img[pts[1]])
		#cap = 10
		#if bgComponents[pts[0]] == bgComponents[pts[1]] and fgComponents[pts[0]] == fgComponents[pts[1]]:
		#if allComponents[pts[0]] == allComponents[pts[1]]:
		#    cap += penaltyForCuttingSameComponent

		beta = 1/(2.0 * 30**2)

		cap = 0
		#cap += .1 * math.exp(-beta * sum([x**2 for x in img[pts[0]] - img[pts[1]]]))
		cap += 1/(edges[pts[0]]+edges[pts[1]] + .001)
			
		assert cap >= 0
		assert cap is not None
		assert not math.isnan(cap)

		"""
                dist = math.sqrt(sum([x**2 for x in img[pts[0]] - img[pts[1]]]))
                cap = 1000.0/(dist + 1)
                """
		#cap = 1000.0/sum([x**2 for x in agResponsibilities[pts[0]] - agResponsibilities[pts[1]]])
		"""
                print(agResponsibilities[pts[0]])
                print(agResponsibilities[pts[1]])
                print(cap)
                exit()
                """

		"""
                differentInFG = fgComponents[pts[0]] != fgComponents[pts[1]]
                fgProb[pts[0]] * fgProb[pts[1]] * differentInFG
                """ 
		binaryCapacities.append(cap)

	return binaryEdges, [x * 2.0 for x in normalizeList(binaryCapacities)]

def initTrimapFromBBox(img, bbox):
	trimap = np.ones(img.shape[:2])
	trimap *= -1

	for x in range(bbox[0], bbox[2]):
		for y in range(bbox[1], bbox[3]):
			trimap[y, x] = 0

	return trimap

def calcMaskUsingMyGrabCut(img, bbox, filename):
	trimap = initTrimapFromBBox(img, bbox)
	pmask = np.zeros(trimap.shape)

	mask = np.copy(trimap)
	mask += 1

	binaryEdges, binaryCapacities = initBinaryEdges(img)

	iteration = 0
	while differenceBetweenTwoMasks(pmask, mask) > .005 or iteration < 3:
		print(differenceBetweenTwoMasks(pmask, mask))

		pmask = np.copy(mask)


		print("Beginning " + filename + " on iteration " + str(iteration))

		fgObs = img[mask == 1]
		bgObs = img[mask == 0]
		allObs = np.asarray(img).reshape(-1, 3)
		"""
            #Plot the point clouds
            #plt.scatter(x=allObs[::100, 2], y=allObs[::100, 0])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            step = 10000

            colorList = [0] * len(fgObs) + [1] * len(bgObs)
            colorArr = np.zeros((len(allObs), 3))

            colorArr[:, 0] = 255 *  np.asarray(colorList)
            colorArr[:, 1] = 255 * (1 - np.asarray(colorList))
            colorArr[:, 2] = 255

            ax.scatter(allObs[::step, 0], allObs[::step, 1], allObs[::step, 2], c=colorArr)

            plt.show()   
            exit()
            """

		numComponents = 10
		covType = 'diag'

		print("Making GMMs...")

		if len(allObs) >= numComponents:
			#Make BG Components
			allGMM = sklearn.mixture.DPGMM(n_components=numComponents, alpha=10, covariance_type=covType, random_state=None, thresh=0.001, min_covar=0.001, n_iter=10, params='wmc', init_params='wmc')    
			allGMM.fit(allObs)

			allComponents = allGMM.predict(np.reshape(img, (-1, 3)))
			allComponents = np.reshape(allComponents, mask.shape)

			agProb = allGMM.score(allObs).reshape(mask.shape)

			agLogProb, agResponsibilities = allGMM.eval(allObs)
			agLogProb = agLogProb.reshape(mask.shape)
			agResponsibilities = agResponsibilities.reshape((mask.shape[0], mask.shape[1], -1))

		if len(fgObs) >= numComponents:
			#Make FG Components
			fgGMM = sklearn.mixture.DPGMM(n_components=numComponents, alpha=.01, covariance_type=covType, random_state=None, thresh=0.001, min_covar=None, n_iter=10, params='wmc', init_params='wmc')
			fgGMM.fit(fgObs)

			fgComponents = fgGMM.predict(np.reshape(img, (-1, 3)))
			fgComponents = np.reshape(fgComponents, mask.shape)

			fgProb = fgGMM.score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)
			#fgProb /= allProb

		if len(bgObs) >= numComponents:
			#Make BG Components
			bgGMM = sklearn.mixture.DPGMM(n_components=numComponents, alpha=.01, covariance_type=covType, random_state=None, thresh=0.001, min_covar=0.001, n_iter=10, params='wmc', init_params='wmc')    
			bgGMM.fit(bgObs)

			bgComponents = bgGMM.predict(np.reshape(img, (-1, 3)))
			bgComponents = np.reshape(bgComponents, mask.shape)

			bgProb = bgGMM.score(np.asarray(img).reshape(-1, 3)).reshape(mask.shape)
			#bgProb /= allProb


		print("We found " + str(np.max(fgComponents) + 1) + " foreground components, and " + str(np.max(bgComponents) + 1) + " background components")

		#Visualize the image mapped to best components of one gaussian mixture model or the other
		agProb = normalizeArr(agProb)
		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "agProb" + ".bmp", agProb)
		fgProb = normalizeArr(fgProb)
		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "fgProb" + ".bmp", fgProb)
		bgProb = normalizeArr(bgProb)
		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "bgProb" + ".bmp", bgProb)



		binaryEdgesArr = np.zeros(mask.shape)
		for edge, cap in zip(binaryEdges, binaryCapacities):
			binaryEdgesArr[edge[0]] += cap
		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "y" + ".bmp", binaryEdgesArr)        

		#Create edges to source and sink
		fgWeight = 1#float(len(fgObs)) / len(allObs)
		bgWeight = 1#float(len(bgObs)) / len(allObs)
		unaryTerm = fgWeight * fgProb - bgWeight * bgProb
		unaryTerm -= np.median(unaryTerm) * .5 #This is the strength of the normalization

		print("Binary edges range from " + str(min(binaryCapacities)) + " to " + str(max(binaryCapacities)) + "\nwith median of " + str(np.median(binaryCapacities)) + " and average of " + str(np.average(binaryCapacities)))
		print("Unary edges range from " + str(np.min(unaryTerm)) + " to " + str(np.max(unaryTerm)) + "\nwith median of " + str(np.median(unaryTerm)) + " and average of " + str(np.average(unaryTerm)))		
		
		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "unaryTerm" + ".bmp", unaryTerm)

		REALLY_BIG_CAPACITY = 10000
		unaryTerm[trimap == -1] = -REALLY_BIG_CAPACITY
		unaryTerm[trimap == 1] = REALLY_BIG_CAPACITY
		
		assert np.min(unaryTerm) < 0
		assert np.max(unaryTerm) > 0

		unaryEdges = []
		unaryCapacities = []
		for x in range(mask.shape[1]):
			for y in range(mask.shape[0]):
				uterm = unaryTerm[y, x]

				if uterm > 0:
					unaryEdges.append(("source", (y,x)))
					unaryCapacities.append(uterm)
				elif uterm < 0:
					unaryEdges.append(((y,x), "sink"))
					unaryCapacities.append(-uterm)

		vertexList = [(y, x) for x in range(mask.shape[1]) for y in range(mask.shape[0])]
		
		assert max(binaryCapacities + unaryCapacities) <= REALLY_BIG_CAPACITY
		assert min(binaryCapacities + unaryCapacities) >= 0
		
		cuts = minCuts(vertexList, binaryEdges + unaryEdges, binaryCapacities + unaryCapacities)
		
		if len(cuts) == 0:
			print("No cuts found! Trying again anyway....")
			print("")
			directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "z" + ".bmp", mask)
			continue

		cut = cuts[0]

		print(cut)

		mask[...] = 0 #zero out the mask
		for vertexIndex in cut[0]:
			if vertexIndex < len(vertexList):
				pt = vertexList[vertexIndex]
				mask[pt] = 1

		directories.saveArrayAsImage(directories.test + filename + "-" + str(iteration) + "z" + ".bmp", mask)
		print("")

		iteration += 1

	return mask

def differenceBetweenTwoMasks(m1, m2):
	assert m1.size == m2.size

	return np.sum(np.abs(m1 - m2))/float(m1.size)

def unaryCostFunction(gmm, pixel):
	prob = gmm.bic(np.asarray([pixel]))
	print(prob)
	#assert 0 <= prob <= 1
	return prob

def calcMaskUsingOpenCVGrabCut(img, bbox):
	mask = np.zeros(img.shape[:2],dtype='uint8')
	tmp1 = np.zeros((1, 13 * 5))
	tmp2 = np.zeros((1, 13 * 5))

	cv2.grabCut(img,mask,bbox,tmp1,tmp2,iterCount=1,mode=cv2.GC_INIT_WITH_RECT)
	return mask

def main():
	directories.clearFolder(directories.test)    

	print("Running GrabCut...")

	for img, bbox, filename in ImagesAndBBoxes:
		bbox = map(int, bbox)
		bbox = tuple(bbox)

		step = 10
		img = sp.misc.imresize(img, float(1)/step)
		bbox = [x/step for x in bbox]

		#mask = calcMaskUsingOpenCVGrabCut(img, bbox)
		mask = calcMaskUsingMyGrabCut(img, bbox, filename)

		#result = cv2.Image(mask)
		print("Finished one image.")
		directories.saveArrayAsImage(directories.output + filename + ".bmp", mask)

	print("GrabCut is finished :D")

if __name__ == "__main__":
	main()