/*
Generate periodic four-dimensional scalar phi-four configurations with local
Metropolis updates.

Inputs: nt, nx, ny, nz, mass-squared, quartic coupling, decorrelation sweeps,
sample count, output path, and optional RNG seed. Output: binary file with dof
and sample-count integers followed by sampled doubles in Eigen column-major
order.
*/
#include <iostream>
#include <random>
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
    int x0, x1, x2, x3;
};

// Simulation parameters for the periodic 4D scalar field.
struct params
{
    int nt, nx, ny, nz, dof;
    double m2, lamda, delta;
    int n_decor, n_thermal, n_conf;
};

// Flatten periodic 4D lattice coordinates into a site index.
inline int Idx(int x0, int x1, int x2, int x3, int nt, int nx, int ny, int nz)
{
    return (x3 % nz) + nz * (x2 % ny) + ny * nz * (x1 % nx) + nx * ny * nz * (x0 % nt);
}

// Convert a flattened site index into its four lattice coordinates.
Index Idx_inv(int n, int nx, int ny, int nz)
{
    struct Index idx;
    idx.x0 = n / (nx * ny * nz);
    idx.x1 = (n - idx.x0 * nx * ny * nz) / (ny * nz);
    idx.x2 = (n - idx.x0 * nx * ny * nz - idx.x1 * ny * nz) / nz;
    idx.x3 = (n - idx.x0 * nx * ny * nz - idx.x1 * ny * nz - idx.x2 * nz) % nz;
    return idx;
}

// Evaluate the potential and nearest-neighbor action terms affected at site n.
double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    struct Index idx;
    idx = Idx_inv(n, p.nx, p.ny, p.nz);

    int idx_mt, idx_mx, idx_my, idx_mz, idx_pt, idx_px, idx_py, idx_pz;
    double pot, kint, kinx, kiny, kinz;

    idx_mt = Idx((idx.x0 - 1 + p.nt) % p.nt, idx.x1, idx.x2, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_mx = Idx(idx.x0, (idx.x1 - 1 + p.nx) % p.nx, idx.x2, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_my = Idx(idx.x0, idx.x1, (idx.x2 - 1 + p.ny) % p.ny, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_mz = Idx(idx.x0, idx.x1, idx.x2, (idx.x3 - 1 + p.nz) % p.nz, p.nt, p.nx, p.ny, p.nz);
    idx_pt = Idx((idx.x0 + 1) % p.nt, idx.x1, idx.x2, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_px = Idx(idx.x0, (idx.x1 + 1) % p.nx, idx.x2, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_py = Idx(idx.x0, idx.x1, (idx.x2 + 1) % p.ny, idx.x3, p.nt, p.nx, p.ny, p.nz);
    idx_pz = Idx(idx.x0, idx.x1, idx.x2, (idx.x3 + 1) % p.nz, p.nt, p.nx, p.ny, p.nz);

    pot = p.m2 / 2.0 * pow(A[n], 2) + p.lamda / 24. * pow(A[n], 4);
    kint = (pow(A[idx_pt] - A[n], 2) + pow(A[idx_mt] - A[n], 2)) / 2.0;
    kinx = (pow(A[idx_px] - A[n], 2) + pow(A[idx_mx] - A[n], 2)) / 2.0;
    kiny = (pow(A[idx_py] - A[n], 2) + pow(A[idx_my] - A[n], 2)) / 2.0;
    kinz = (pow(A[idx_pz] - A[n], 2) + pow(A[idx_mz] - A[n], 2)) / 2.0;

    return pot + kint + kinx + kiny + kinz;
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

// Collect n_conf configurations, separated by n_decor full Metropolis sweeps.
Eigen::MatrixXd Sweep(Eigen::ArrayXd &A, params &p)
{
    return mc::sweep(A, p, p.n_decor, Metropolis);
}

// Evolve the field for the configured number of thermalization sweeps.
Eigen::ArrayXd Thermalization(Eigen::ArrayXd &A, params &p)
{
    return mc::thermalize(A, p, p.dof, p.n_thermal, Metropolis);
}

// Tune the proposal width until the measured acceptance fraction is in range.
Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    return mc::calibrate(A, p, p.dof, Metropolis, random);
}

// Parse lattice/coupling/sample arguments and write header plus configuration data.
int main(int argc, char **argv)
{

    struct params p;
    p.delta = 1;

    unsigned int seed = 42;
    std::string output_path;
    CLI::App app{"Generate periodic 4D scalar phi-four configurations"};
    app.add_option("--nt", p.nt, "Temporal lattice extent")->required();
    app.add_option("--nx", p.nx, "First spatial lattice extent")->required();
    app.add_option("--ny", p.ny, "Second spatial lattice extent")->required();
    app.add_option("--nz", p.nz, "Third spatial lattice extent")->required();
    app.add_option("--mass-squared", p.m2, "Scalar mass squared")->required();
    app.add_option("--lambda", p.lamda, "Quartic coupling")->required();
    app.add_option("--decorrelation-sweeps", p.n_decor, "Sweeps between samples")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

    p.dof = p.nt * p.nx * p.ny * p.nz;
    random.reseed(seed);

    Eigen::ArrayXd configuration = Eigen::ArrayXd::Zero(p.dof); // Cold start

    Calibrate(configuration, p);
    Thermalization(configuration, p);
    Calibrate(configuration, p);
    Eigen::MatrixXd sample = Sweep(configuration, p);

    if (!mc::write_samples(output_path, p.dof, p.n_conf, sample))
    {
        std::cerr << "Error writing sample file.\n";
        return 1;
    }

    return 0;
}