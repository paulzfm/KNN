targets : contest knn

contest: contest.cu
	nvcc contest.cu -o contest

knn: knn.cu
	nvcc knn.cu -o knn
