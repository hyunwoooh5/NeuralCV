#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include <ctime> // Timer
#include <stdio.h>

typedef std::complex<double> dcomp;
const dcomp I(0, 1);
const double PI = std::atan(1.0) * 4;

// Random Number Generator
// std::default_random_engine generator; // for random engine reset
std::random_device generator;                           // get non-deterministic(truly random) seed
std::mt19937 gen(generator());                          // reset RNG
std::uniform_real_distribution<double> dist(-1.0, 1.0); // -1.0 to 1.0 uniform distribution
std::uniform_real_distribution<> rand01(0.0, 1.0);      // For Metropolis

int accept = 0; // For acceptance rate, Should not be defined again

// Functions
struct Index
{
    int x0, x1, x2;
};

struct params
{
    int nt, nx, ny, dof;
    double m2, lamda, delta;
    int n_decor, n_thermal, n_conf;
};

inline int Idx(int x0, int x1, int x2, int nt, int nx, int ny)
{
    return (x2 % ny) + ny * (x1 % nx) + nx * ny * (x0 % nt);
}

Index Idx_inv(int n, int nx, int ny)
{
    struct Index idx;
    idx.x0 = n / (nx * ny);
    idx.x1 = (n - idx.x0 * nx * ny) / ny;
    idx.x2 = (n - idx.x0 * nx * ny - idx.x1 * nx) % ny;
    return idx;
}

double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    struct Index idx;
    idx = Idx_inv(n, p.nx, p.ny);

    int idx_mt, idx_mx, idx_my, idx_pt, idx_px, idx_py;
    double pot, kint, kinx, kiny;

    idx_mt = Idx((idx.x0 - 1 + p.nt) % p.nt, idx.x1, idx.x2, p.nt, p.nx, p.ny);
    idx_mx = Idx(idx.x0, (idx.x1 - 1 + p.nx) % p.nx, idx.x2, p.nt, p.nx, p.ny);
    idx_my = Idx(idx.x0, idx.x1, (idx.x2 - 1 + p.ny) % p.ny, p.nt, p.nx, p.ny);
    idx_pt = Idx((idx.x0 + 1) % p.nt, idx.x1, idx.x2, p.nt, p.nx, p.ny);
    idx_px = Idx(idx.x0, (idx.x1 + 1) % p.nx, idx.x2, p.nt, p.nx, p.ny);
    idx_py = Idx(idx.x0, idx.x1, (idx.x2 + 1) % p.ny, p.nt, p.nx, p.ny);

    pot = p.m2 / 2.0 * pow(A[n], 2) + p.lamda / 24. * pow(A[n], 4);
    kint = (pow(A[idx_pt] - A[n], 2) + pow(A[idx_mt] - A[n], 2)) / 2.0;
    kinx = (pow(A[idx_px] - A[n], 2) + pow(A[idx_mx] - A[n], 2)) / 2.0;
    kiny = (pow(A[idx_py] - A[n], 2) + pow(A[idx_my] - A[n], 2)) / 2.0;

    return pot + kint + kinx + kiny;
}

Eigen::ArrayXd Metropolis(Eigen::ArrayXd &A, int n, params &p)
{
    Eigen::ArrayXd A_new = A;
    A_new[n] += p.delta * dist(gen);
    double dS = Action_Local(A_new, n, p) - Action_Local(A, n, p);

    if (exp(-dS) >= rand01(gen))
    {
        accept++;

        return A_new;
    }
    else
    {
        return A;
    }
}

Eigen::MatrixXd Sweep(Eigen::ArrayXd &A, params &p)
{
    Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(p.dof, p.n_conf);

    for (int i = 0; i < p.n_conf; i++)
    {
        for (int j = 0; j < p.n_decor; j++)
        {
            for (int k = 0; k < p.dof; k++)
            {
                A = Metropolis(A, k, p);
            }
        }
        samples.col(i) = A;
    }

    return samples;
}

Eigen::ArrayXd Thermalization(Eigen::ArrayXd &A, params &p)
{
    for (int i = 0; i < p.n_thermal; i++)
    {
        for (int j = 0; j < p.dof; j++)
        {
            A = Metropolis(A, j, p);
        }
    }

    return A;
}

Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    double ratio = 0;
    while (ratio <= 0.3 || ratio >= 0.55)
    {
        accept = 0;
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < p.dof; j++)
            {
                A = Metropolis(A, j, p);
            }
        }
        ratio = (double)accept / (p.dof * 10);
        if (ratio >= 0.55)
        {
            p.delta = p.delta * 1.02;
        }
        else if (ratio <= 0.3)
        {
            p.delta = p.delta * 0.98;
        }
    }

    return A;
}

int main(int argc, char **argv)
{

    struct params p;
    p.delta = 1;

    p.nt = std::stoi(argv[1]);
    p.nx = std::stoi(argv[2]);
    p.ny = std::stoi(argv[3]);
    p.dof = p.nt * p.nx * p.ny;
    p.m2 = std::stod(argv[4]);
    p.lamda = std::stod(argv[5]);
    p.n_decor = std::stoi(argv[6]);
    p.n_thermal = 10000;
    p.n_conf = std::stoi(argv[7]);

    
    Eigen::ArrayXd configuration = Eigen::ArrayXd::Zero(p.dof); // Cold start

    Calibrate(configuration, p);
    Thermalization(configuration, p);
    Calibrate(configuration, p);
    Eigen::MatrixXd sample = Sweep(configuration, p);

    std::ofstream outfile(argv[8], std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Error opening file for writing.\n";
        return 1;
    }

    outfile.write(reinterpret_cast<char *>(&p.dof), sizeof(int));
    outfile.write(reinterpret_cast<char *>(&p.n_conf), sizeof(int));

    // Write the Eigen array data to the file in binary format
    outfile.write(reinterpret_cast<const char *>(sample.data()), sample.size() * sizeof(double));

    // Close the file
    outfile.close();

    return 0;
}