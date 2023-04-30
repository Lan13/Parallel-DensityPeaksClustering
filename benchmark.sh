g++ seq_dpc.cpp -o seq_dpc
g++ omp_dpc.cpp -o omp_dpc -fopenmp
nvcc cu_dpc.cu -o cu_dpc
nvcc cu_dpc2.cu -o cu_dpc2

input='./catn.txt'
num='31'

seqTime=$(./seq_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
ompTime=$(./omp_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cudaTime=$(./cu_dpc -n ${num} -i ${input} | grep 'time' | awk '{print $4}')
cuda2Time=$(./cu_dpc2 -n ${num} -i ${input} | grep 'time' | awk '{print $4}')

echo ""
echo "===================== DensityPeaksClustering ========================"
echo "---------------------------------------------------------------------"
speedup=$(echo "scale=2; ${seqTime} / ${ompTime}" | bc)
echo "seqTime = ${seqTime}ms  ompTime = ${ompTime}ms  speedup = ${speedup}x"
speedup=$(echo "scale=2; ${seqTime} / ${cudaTime}" | bc)
echo "seqTime = ${seqTime}ms  cudaTime = ${cudaTime}ms  speedup = ${speedup}x"
speedup=$(echo "scale=2; ${seqTime} / ${cuda2Time}" | bc)
echo "seqTime = ${seqTime}ms  cuda2Time = ${cuda2Time}ms  speedup = ${speedup}x"
echo "---------------------------------------------------------------------"
echo ""
