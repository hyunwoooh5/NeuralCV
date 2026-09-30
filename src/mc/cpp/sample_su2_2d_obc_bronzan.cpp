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
struct params
{
    int dof;
    double g, delta;
    int n_thermal, n_conf;
};

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
double Action(Eigen::ArrayXd &A, params &p)
{
    if (A[0] < 0 || A[0] > PI / 2)
    {
        return pow(10, 10);
    }
    else
    {
        return -4. / pow(p.g, 2) * sin(A[0]) * cos(A[1]) - log(sin(2 * A[0]));
    }
}

Eigen::ArrayXd Metropolis(Eigen::ArrayXd &A, params &p)
{
    Eigen::ArrayXd A_new = A + p.delta * Eigen::ArrayXd::NullaryExpr(p.dof, [&]()
                                                                     { return dist(gen); });

    double dS = Action(A_new, p) - Action(A, p);

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
    for (int i = 0; i < p.dof * p.n_thermal; i++)
    {
        A = Metropolis(A, p);
    }

    return A;
}

Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    double ratio = 0;
    while (ratio <= 0.3 || ratio >= 0.55)
    {
        accept = 0;
        for (int i = 0; i < 10 * p.dof; i++)
        {
            A = Metropolis(A, p);
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
    p.dof = pow(2, 2) - 1;

    p.g = std::stod(argv[1]);
    p.n_thermal = 10000;
    p.n_conf = std::stoi(argv[2]);

    Eigen::ArrayXd configuration = Eigen::ArrayXd::Zero(p.dof); // Cold start

    Calibrate(configuration, p);
    Thermalization(configuration, p);
    Calibrate(configuration, p);
    Eigen::MatrixXd sample = Sweep(configuration, p);

    std::ofstream outfile(argv[3], std::ios::binary);
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