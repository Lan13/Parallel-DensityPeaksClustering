g++ seq_dpc.cpp -o seq_dpc
g++ omp_dpc.cpp -o omp_dpc -fopenmp
g++ omp_dpc2.cpp -o omp_dpc2 -fopenmp
nvcc cu_dpc.cu -o cu_dpc
nvcc cu_dpc2.cu -o cu_dpc2
nvcc cu_dpc3.cu -o cu_dpc3
nvcc cu_dpc4.cu -o cu_dpc4

input='./catn.txt'
num='31'

seqTime=$(./seq_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
ompTime=$(./omp_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
omp2Time=$(./omp_dpc2 -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cudaTime=$(./cu_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cuda2Time=$(./cu_dpc2 -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cuda3Time=$(./cu_dpc3 -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cuda4Time=$(./cu_dpc4 -n ${num} -i ${input} | grep 'time' | awk '{print $4}')

echo ""
echo "========================== DensityPeaksClustering ============================="
echo "-------------------------------------------------------------------------------"
speedup=$(echo "scale=2; ${seqTime} / ${ompTime}" | bc)
echo "seqTime = ${seqTime}ms  ompTime = ${ompTime}ms  speedup = ${speedup}x  (OpenMP Version)"
speedup=$(echo "scale=2; ${seqTime} / ${omp2Time}" | bc)
echo "seqTime = ${seqTime}ms  omp2Time = ${omp2Time}ms  speedup = ${speedup}x  (Argsort Version)"
speedup=$(echo "scale=2; ${seqTime} / ${cudaTime}" | bc)
echo "seqTime = ${seqTime}ms  cudaTime = ${cudaTime}ms  speedup = ${speedup}x  (Cuda Version)"
speedup=$(echo "scale=2; ${seqTime} / ${cuda2Time}" | bc)
echo "seqTime = ${seqTime}ms  cuda2Time = ${cuda2Time}ms  speedup = ${speedup}x  (Optimized Version)"
speedup=$(echo "scale=2; ${seqTime} / ${cuda3Time}" | bc)
echo "seqTime = ${seqTime}ms  cuda3Time = ${cuda3Time}ms  speedup = ${speedup}x  (Texture Version)"
speedup=$(echo "scale=2; ${seqTime} / ${cuda4Time}" | bc)
echo "seqTime = ${seqTime}ms  cuda4Time = ${cuda4Time}ms  speedup = ${speedup}x  (Shared Version)"
echo "-------------------------------------------------------------------------------"
echo ""
