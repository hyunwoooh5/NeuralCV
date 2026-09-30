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
    int x0, x1;
};

struct params
{
    int nt, nx, dof;
    double beta, delta;
    int n_decor, n_thermal, n_conf;
};

inline int Idx(int x0, int x1, int nt, int nx)
{
    return (x1 % nx) + nx * (x0 % nt);
}

Index Idx_inv(int n, int nx)
{
    struct Index idx;
    idx.x0 = n / nx;
    idx.x1 = n % nx;
    return idx;
}

dcomp Log_Det(const Eigen::MatrixXcd &m, Eigen::MatrixXcd *inv = NULL)
{
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(m); // LU decomposition of M
    dcomp res = 0;
    for (int i = 0; i < m.col(0).size(); ++i)
        res += log(lu.matrixLU()(i, i)); // Calculating LogDet

    res += (lu.permutationP().determinant() == -1) ? I * PI : 0.0;
    res -= I * 2.0 * PI * round(res.imag() / (2.0 * PI));
    if (inv != NULL)
        *inv = lu.inverse();
    return res;
}

// Metropolis
double Action(Eigen::ArrayXd &A)
{
    return 0;
}

double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    return -p.beta*cos(A[n]);
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
    p.dof = p.nt * p.nx;
    p.beta = std::stod(argv[3]);
    p.n_decor = std::stoi(argv[4]);
    p.n_thermal = 10000;
    p.n_conf = std::stoi(argv[5]);

    Eigen::ArrayXd configuration = Eigen::ArrayXd::Zero(p.dof); // Cold start

    Calibrate(configuration, p);
    Thermalization(configuration, p);
    Calibrate(configuration, p);
    Eigen::MatrixXd sample = Sweep(configuration, p);

    std::ofstream outfile(argv[6], std::ios::binary);
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