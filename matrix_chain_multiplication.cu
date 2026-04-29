#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <climits>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// Flatten 2D index
__host__ __device__ inline int idx(int i, int j, int n)
{
    return i * n + j;
}

// ---------------- CPU VERSION ----------------
long long mcmCPU(const vector<int> &dims)
{
    int n = dims.size() - 1;
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    for (int len = 2; len <= n; len++)
    {
        for (int i = 0; i + len - 1 < n; i++)
        {
            int j = i + len - 1;
            dp[i][j] = LLONG_MAX;

            for (int k = i; k < j; k++)
            {
                long long cost = dp[i][k] + dp[k + 1][j] +
                                 1LL * dims[i] * dims[k + 1] * dims[j + 1];

                dp[i][j] = min(dp[i][j], cost);
            }
        }
    }

    return dp[0][n - 1];
}

// ---------------- GPU KERNEL ----------------
__global__ void mcmKernel(int *dims, long long *dp, int n, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i + len - 1;

    if (i >= n || j >= n)
        return;

    long long best = LLONG_MAX;

    for (int k = i; k < j; k++)
    {
        long long left = dp[idx(i, k, n)];
        long long right = dp[idx(k + 1, j, n)];
        long long cost = left + right +
                         1LL * dims[i] * dims[k + 1] * dims[j + 1];

        if (cost < best)
            best = cost;
    }

    dp[idx(i, j, n)] = best;
}

// ---------------- GPU FUNCTION ----------------
long long mcmGPU(const vector<int> &dims, float &gpuTime)
{
    int n = dims.size() - 1;

    int *d_dims;
    long long *d_dp;

    size_t dimsSize = dims.size() * sizeof(int);
    size_t dpSize = n * n * sizeof(long long);

    cudaMalloc(&d_dims, dimsSize);
    cudaMalloc(&d_dp, dpSize);

    cudaMemcpy(d_dims, dims.data(), dimsSize, cudaMemcpyHostToDevice);
    cudaMemset(d_dp, 0, dpSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int len = 2; len <= n; len++)
    {
        int total = n - len + 1;

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        mcmKernel<<<blocks, threads>>>(d_dims, d_dp, n, len);

        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);

    long long result;
    cudaMemcpy(&result, &d_dp[idx(0, n - 1, n)], sizeof(long long), cudaMemcpyDeviceToHost);

    cudaFree(d_dims);
    cudaFree(d_dp);

    return result;
}

// ---------------- BENCHMARK ----------------
int main()
{
    vector<int> testSizes = {50, 100, 200, 300};

    random_device rd;
    mt19937 gen(rd());

    // 🔥 Increased range for heavier computation
    uniform_int_distribution<int> dist(10, 500);

    cout << left << setw(8) << "N"
         << setw(15) << "CPU(ms)"
         << setw(15) << "GPU(ms)"
         << setw(12) << "Speedup" << endl;

    cout << string(50, '-') << endl;

    for (int n : testSizes)
    {
        vector<int> dims(n + 1);
        for (int i = 0; i <= n; i++)
            dims[i] = dist(gen);

        // CPU timing
        auto start = chrono::high_resolution_clock::now();
        long long cpuRes = mcmCPU(dims);
        auto end = chrono::high_resolution_clock::now();

        double cpuTime = chrono::duration<double, milli>(end - start).count();

        // GPU timing
        float gpuTime = 0;
        long long gpuRes = mcmGPU(dims, gpuTime);

        double speedup = cpuTime / gpuTime;

        cout << left << setw(8) << n
             << setw(15) << fixed << setprecision(3) << cpuTime
             << setw(15) << gpuTime
             << setw(12) << speedup;

        if (cpuRes == gpuRes)
            cout << "OK";
        else
            cout << "Mismatch";

        cout << endl;
    }

    return 0;
}