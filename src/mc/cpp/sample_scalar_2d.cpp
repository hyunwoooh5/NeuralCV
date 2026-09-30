/*
Generate periodic two-dimensional scalar phi-four configurations with local
Metropolis updates.

Inputs: nt, nx, mass-squared, quartic coupling, decorrelation sweeps, sample
count, output path, and optional RNG seed. Output: binary file with dof and
sample-count integers, followed by the sampled doubles in Eigen column-major
order.
*/
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <ctime> // Timer
#include <stdio.h>
#include "monte_carlo.hpp"

typedef std::complex<double> dcomp;
const dcomp I(0, 1);
const double PI = std::atan(1.0) * 4;

mc::Random random;

// Functions
// Lattice coordinates for one scalar-field site.
struct Index
{
    int x0, x1;
};

// Simulation parameters for the periodic 2D scalar field.
struct params
{
    int nt, nx, dof;
    double m2, lamda, delta;
    int n_decor, n_thermal, n_conf;
};

// Flatten periodic lattice coordinates into a site index.
inline int Idx(int x0, int x1, int nt, int nx)
{
    return (x1 % nx) + nx * (x0 % nt);
}

// Convert a flattened site index into its two lattice coordinates.
Index Idx_inv(int n, int nx)
{
    struct Index idx;
    idx.x0 = n / nx;
    idx.x1 = n % nx;
    return idx;
}

// Return the complex log determinant and optionally store the matrix inverse.
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
// Placeholder for a full-action calculation; the sampler uses Action_Local.
double Action(Eigen::ArrayXd &A)
{
    return 0;
}

// Evaluate the potential and nearest-neighbor action terms affected at site n.
double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    struct Index idx;
    idx = Idx_inv(n, p.nx);

    int idx_mt, idx_mx, idx_pt, idx_px;
    double pot, kint, kinx;

    idx_mt = Idx((idx.x0 - 1 + p.nt) % p.nt, idx.x1, p.nt, p.nx);
    idx_mx = Idx(idx.x0, (idx.x1 - 1 + p.nx) % p.nx, p.nt, p.nx);
    idx_pt = Idx((idx.x0 + 1) % p.nt, idx.x1, p.nt, p.nx);
    idx_px = Idx(idx.x0, (idx.x1 + 1) % p.nx, p.nt, p.nx);

    pot = p.m2 / 2.0 * pow(A[n], 2) + p.lamda / 24. * pow(A[n], 4);
    kint = (pow(A[idx_pt] - A[n], 2) + pow(A[idx_mt] - A[n], 2)) / 2.0;
    kinx = (pow(A[idx_px] - A[n], 2) + pow(A[idx_mx] - A[n], 2)) / 2.0;

    return pot + kint + kinx;
}

// Propose and accept/reject a change to one scalar-field site.
Eigen::ArrayXd Metropolis(Eigen::ArrayXd &A, int n, params &p)
{
    Eigen::ArrayXd A_new = A;
    A_new[n] += p.delta * random.proposal();
    double dS = Action_Local(A_new, n, p) - Action_Local(A, n, p);

    if (random.accept(dS))
    {
        return A_new;
    }
    else
    {
        return A;
    }
}

// Parse lattice/coupling/sample arguments and write header plus configuration data.
int main(int argc, char **argv)
{
    struct params p;
    p.delta = 1;

    unsigned int seed = 42;
    std::string output_path;
    CLI::App app{"Generate periodic 2D scalar phi-four configurations"};
    app.add_option("--nt", p.nt, "Temporal lattice extent")->required();
    app.add_option("--nx", p.nx, "Spatial lattice extent")->required();
    app.add_option("--mass-squared", p.m2, "Scalar mass squared")->required();
    app.add_option("--lambda", p.lamda, "Quartic coupling")->required();
    app.add_option("--decorrelation-sweeps", p.n_decor, "Sweeps between samples")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

    p.dof = p.nt * p.nx;
    random.reseed(seed);

    Eigen::ArrayXd configuration = Eigen::ArrayXd::Zero(p.dof); // Cold start

    mc::calibrate(configuration, p, p.dof, Metropolis, random);
    mc::thermalize(configuration, p, p.dof, p.n_thermal, Metropolis);
    mc::calibrate(configuration, p, p.dof, Metropolis, random);
    Eigen::MatrixXd sample = mc::sweep(configuration, p, p.n_decor, Metropolis);

    if (!mc::write_samples(output_path, p.dof, p.n_conf, sample))
    {
        std::cerr << "Error writing sample file.\n";
        return 1;
    }

    return 0;
}