/*
Generate two-dimensional open-boundary U(1) gauge configurations with local
Metropolis updates.

Inputs: nt, nx, coupling beta, decorrelation sweeps, sample count, output path,
and optional RNG seed. Output: binary file with dof and sample-count integers
followed by sampled doubles in Eigen column-major order.
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
// Lattice coordinates for one site.
struct Index
{
    int x0, x1;
};

// Simulation parameters for the open-boundary 2D U(1) field.
struct params
{
    int nt, nx, dof;
    double beta, delta;
    int n_decor, n_thermal, n_conf;
};

// Flatten a site coordinate on the 2D lattice.
inline int Idx(int x0, int x1, int nt, int nx)
{
    return (x1 % nx) + nx * (x0 % nt);
}

// Convert a flattened site index to lattice coordinates.
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
// Placeholder for the full action; updates use the local cosine action below.
double Action(Eigen::ArrayXd &A)
{
    return 0;
}

// Evaluate the local open-boundary U(1) action contribution for link n.
double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    return -p.beta*cos(A[n]);
}

// Propose and accept/reject an update to link angle n.
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

// Collect n_conf configurations separated by n_decor Metropolis sweeps.
Eigen::MatrixXd Sweep(Eigen::ArrayXd &A, params &p)
{
    return mc::sweep(A, p, p.n_decor, Metropolis);
}

// Apply the configured number of thermalization sweeps to the state.
Eigen::ArrayXd Thermalization(Eigen::ArrayXd &A, params &p)
{
    return mc::thermalize(A, p, p.dof, p.n_thermal, Metropolis);
}

// Tune the proposal width until the measured acceptance fraction is in range.
Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    return mc::calibrate(A, p, p.dof, Metropolis, random);
}

// Parse lattice/coupling/sample arguments and write header plus sampled angles.
int main(int argc, char **argv)
{
    struct params p;
    p.delta = 1;

    unsigned int seed = 42;
    std::string output_path;
    CLI::App app{"Generate 2D open-boundary U(1) configurations"};
    app.add_option("--nt", p.nt, "Temporal lattice extent")->required();
    app.add_option("--nx", p.nx, "Spatial lattice extent")->required();
    app.add_option("--beta", p.beta, "Gauge coupling")->required();
    app.add_option("--decorrelation-sweeps", p.n_decor, "Sweeps between samples")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

    p.dof = p.nt * p.nx;
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