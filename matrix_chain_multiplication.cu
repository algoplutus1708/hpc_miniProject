#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <climits>
#include <cuda_runtime.h>

using namespace std;

__host__ __device__ inline int dpIndex(int row, int col, int n)
{
    return row * n + col;
}

bool checkCuda(cudaError_t status, const char *message)
{
    if (status != cudaSuccess)
    {
        cerr << message << ": " << cudaGetErrorString(status) << '\n';
        return false;
    }

    return true;
}

long long matrixChainMultiplicationCPU(const vector<int> &dims)
{
    int n = static_cast<int>(dims.size()) - 1;

    // dp[i][j] stores the minimum cost to multiply matrices i through j.
    // Matrices are numbered from 0 to n - 1.
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    // len is the chain length.
    // Start from chains of length 2 and build up to the full chain.
    for (int len = 2; len <= n; ++len)
    {
        for (int i = 0; i + len - 1 < n; ++i)
        {
            int j = i + len - 1;
            dp[i][j] = numeric_limits<long long>::max();

            // Try every possible split point.
            for (int k = i; k < j; ++k)
            {
                long long cost = dp[i][k] + dp[k + 1][j] +
                                 1LL * dims[i] * dims[k + 1] * dims[j + 1];
                if (cost < dp[i][j])
                {
                    dp[i][j] = cost;
                }
            }
        }
    }

    return dp[0][n - 1];
}

// Each kernel launch computes one anti-diagonal of the DP table.
// Every thread owns exactly one cell dp[i][j] on that diagonal.
// The split point k is still scanned sequentially inside the thread,
// which matches the matrix-chain recurrence and keeps the logic simple.
__global__ void matrixChainMultiplicationKernel(const int *dims, long long *dp, int n, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i + len - 1;

    // Only threads that map to a valid dp[i][j] cell should do work.
    if (i >= n || j >= n)
    {
        return;
    }

    long long best = LLONG_MAX;

    // Try every possible split point for the current subchain.
    // The shorter subproblems dp[i][k] and dp[k+1][j] were computed by
    // earlier kernel launches, so they are already available in global memory.
    for (int k = i; k < j; ++k)
    {
        long long leftCost = dp[dpIndex(i, k, n)];
        long long rightCost = dp[dpIndex(k + 1, j, n)];
        long long multiplyCost = 1LL * dims[i] * dims[k + 1] * dims[j + 1];
        long long currentCost = leftCost + rightCost + multiplyCost;

        if (currentCost < best)
        {
            best = currentCost;
        }
    }

    dp[dpIndex(i, j, n)] = best;
}

long long matrixChainMultiplicationGPU(const vector<int> &dims)
{
    int n = static_cast<int>(dims.size()) - 1;
    size_t dimsBytes = dims.size() * sizeof(int);
    size_t dpBytes = static_cast<size_t>(n) * n * sizeof(long long);

    int *deviceDims = nullptr;
    long long *deviceDp = nullptr;
    long long minimumCost = -1;

    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceDims), dimsBytes),
                   "Failed to allocate device dimensions"))
    {
        return -1;
    }

    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceDp), dpBytes),
                   "Failed to allocate device DP table"))
    {
        cudaFree(deviceDims);
        return -1;
    }

    // The DP table lives in global memory on the GPU.
    // Zero-initialization is enough because the diagonal entries dp[i][i] are 0.
    if (!checkCuda(cudaMemset(deviceDp, 0, dpBytes), "Failed to initialize device DP table"))
    {
        cudaFree(deviceDp);
        cudaFree(deviceDims);
        return -1;
    }

    if (!checkCuda(cudaMemcpy(deviceDims, dims.data(), dimsBytes, cudaMemcpyHostToDevice),
                   "Failed to copy dimensions to device"))
    {
        cudaFree(deviceDp);
        cudaFree(deviceDims);
        return -1;
    }

    // Build the table diagonal by diagonal.
    // For each chain length, launch one kernel and let each thread compute one cell.
    for (int len = 2; len <= n; ++len)
    {
        int cellsOnDiagonal = n - len + 1;
        int threadsPerBlock = 256;
        int blocks = (cellsOnDiagonal + threadsPerBlock - 1) / threadsPerBlock;

        matrixChainMultiplicationKernel<<<blocks, threadsPerBlock>>>(deviceDims, deviceDp, n, len);

        if (!checkCuda(cudaGetLastError(), "Kernel launch failed"))
        {
            cudaFree(deviceDp);
            cudaFree(deviceDims);
            return -1;
        }

        // Wait for this diagonal to finish before starting the next one.
        if (!checkCuda(cudaDeviceSynchronize(), "Kernel execution failed"))
        {
            cudaFree(deviceDp);
            cudaFree(deviceDims);
            return -1;
        }
    }

    if (!checkCuda(cudaMemcpy(&minimumCost, deviceDp + dpIndex(0, n - 1, n), sizeof(long long), cudaMemcpyDeviceToHost),
                   "Failed to copy result back to host"))
    {
        cudaFree(deviceDp);
        cudaFree(deviceDims);
        return -1;
    }

    cudaFree(deviceDp);
    cudaFree(deviceDims);
    return minimumCost;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cout << "Enter number of matrices: ";
    cin >> n;

    if (n <= 0)
    {
        cout << "Number of matrices must be positive.\n";
        return 0;
    }

    auto start = chrono::high_resolution_clock::now();

    // Generate a random dimensions array of size n + 1.
    // If there are n matrices, we need n + 1 dimension values.
    vector<int> dims(n + 1);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(2, 10);

    for (int i = 0; i <= n; ++i)
    {
        dims[i] = dist(gen);
    }

    cout << "Generated dimensions: ";
    for (int i = 0; i <= n; ++i)
    {
        cout << dims[i] << (i == n ? '\n' : ' ');
    }

    long long minimumCost = matrixChainMultiplicationGPU(dims);
    if (minimumCost < 0)
    {
        cout << "GPU execution failed, falling back to CPU version.\n";
        minimumCost = matrixChainMultiplicationCPU(dims);
    }

    cout << "Minimum multiplication cost: " << minimumCost << '\n';

    auto end = chrono::high_resolution_clock::now();
    auto executionTime = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << executionTime.count() << " ms\n";

    return 0;
}
