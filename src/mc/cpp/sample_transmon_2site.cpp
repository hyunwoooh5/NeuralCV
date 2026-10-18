/*
Generate periodic two-site transmon phase configurations with local Metropolis
updates.

Inputs: time-slice count Nt, total time t, charging energies E_C1/E_C2,
Josephson energies E_J1/E_J2, coupling energy E_CC, decorrelation sweeps,
sample count, output path, and optional RNG seed.
Output: binary file with dof and sample-count integers followed by sampled
doubles in Eigen column-major order, with each configuration ordered by time
slice and then site.
*/
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>
#include "monte_carlo.hpp"

const double PI = std::atan(1.0) * 4;

mc::Random rng;

// Physical parameters and Monte Carlo controls for the two-site transmon.
struct params
{
    int nt, dof;
    double E_C1, E_J1, E_C2, E_J2, E_CC, t, dt, delta;
    double C1, C2, CC;

    int n_decor, n_thermal, n_conf;
};

// Lattice coordinates for one phase component.
struct Index
{
    int t, site;
};

// Flatten time and site coordinates using the Python (Nt, 2) ordering.
inline int Idx(int t, int site, int nt)
{
    return 2 * (t % nt) + site;
}

// Convert a flattened phase-component index into time and site coordinates.
Index Idx_inv(int n)
{
    Index idx;
    idx.t = n / 2;
    idx.site = n % 2;
    return idx;
}

// Evaluate the complete periodic two-site transmon action.
double Action(Eigen::ArrayXd &A, params &p)
{
    double kin = 0.0;
    double pot = 0.0;

    for (int t = 0; t < p.nt; ++t)
    {
        const int next = (t + 1) % p.nt;
        const double delta1 = A[Idx(next, 0, p.nt)] - A[Idx(t, 0, p.nt)];
        const double delta2 = A[Idx(next, 1, p.nt)] - A[Idx(t, 1, p.nt)];

        kin += p.C1 * delta1 * delta1 + p.C2 * delta2 * delta2
               + p.CC * (delta1 - delta2) * (delta1 - delta2);
        pot -= p.E_J1 * std::cos(A[Idx(t, 0, p.nt)]) * p.dt;
        pot -= p.E_J2 * std::cos(A[Idx(t, 1, p.nt)]) * p.dt;
    }

    return kin / (8.0 * p.dt) + pot;
}

// Evaluate the potential and two neighboring temporal links affected at n.
double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    const Index idx = Idx_inv(n);
    const int previous = (idx.t - 1 + p.nt) % p.nt;
    const int next = (idx.t + 1) % p.nt;

    const double forward1 = A[Idx(next, 0, p.nt)] - A[Idx(idx.t, 0, p.nt)];
    const double forward2 = A[Idx(next, 1, p.nt)] - A[Idx(idx.t, 1, p.nt)];
    const double backward1 = A[Idx(idx.t, 0, p.nt)] - A[Idx(previous, 0, p.nt)];
    const double backward2 = A[Idx(idx.t, 1, p.nt)] - A[Idx(previous, 1, p.nt)];

    const double kin = p.C1 * (forward1 * forward1 + backward1 * backward1)
                       + p.C2 * (forward2 * forward2 + backward2 * backward2)
                       + p.CC * ((forward1 - forward2) * (forward1 - forward2)
                                 + (backward1 - backward2) * (backward1 - backward2));
    const double E_J = idx.site == 0 ? p.E_J1 : p.E_J2;
    const double pot = -E_J * std::cos(A[n]) * p.dt;

    return kin / (8.0 * p.dt) + pot;
}


// Propose and accept/reject a local phase change at component n.
Eigen::ArrayXd Metropolis(Eigen::ArrayXd &A, int n, params &p)
{
    Eigen::ArrayXd A_new = A;
    A_new[n] += p.delta * rng.proposal();
    double dS = Action_Local(A_new, n, p) - Action_Local(A, n, p);

    if (rng.accept(dS))
    {
        return A_new;
    }
    else
    {
        return A;
    }
}

// Collect n_conf phase configurations separated by n_decor sweeps.
Eigen::MatrixXd Sweep(Eigen::ArrayXd &A, params &p)
{
    return mc::sweep(A, p, p.n_decor, Metropolis);
}

// Apply the configured number of thermalization sweeps to the phase state.
Eigen::ArrayXd Thermalization(Eigen::ArrayXd &A, params &p)
{
    return mc::thermalize(A, p, p.dof, p.n_thermal, Metropolis);
}

// Tune the proposal width until the measured acceptance fraction is in range.
Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    return mc::calibrate(A, p, p.dof, Metropolis, rng);
}

// Parse transmon parameters and write a binary configuration file.
int main(int argc, char **argv)
{
    struct params p;
    p.delta = 1;

    unsigned int seed = 42;
    std::string output_path;
    CLI::App app{"Generate periodic two-site transmon phase configurations"};
    app.add_option("--nt", p.nt, "Number of time slices")
        ->required()->check(CLI::PositiveNumber);
    app.add_option("--t", p.t, "Total simulation time in nanoseconds")->required();
    app.add_option("--E_C1", p.E_C1,
                   "Site 1 charging energy in GHz")->required();
    app.add_option("--E_J1", p.E_J1,
                   "Site 1 Josephson energy in GHz")->required();
    app.add_option("--E_C2", p.E_C2,
                   "Site 2 charging energy in GHz")->required();
    app.add_option("--E_J2", p.E_J2,
                   "Site 2 Josephson energy in GHz")->required();
    app.add_option("--E_CC", p.E_CC,
                   "Velocity coupling energy in GHz")->required();
    app.add_option("--sweeps", p.n_decor, "Sweeps between samples")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

    p.dof = 2 * p.nt;
    p.E_C1 *= 2 * PI * 1e9;
    p.E_J1 *= 2 * PI * 1e9;
    p.E_C2 *= 2 * PI * 1e9;
    p.E_J2 *= 2 * PI * 1e9;
    p.E_CC *= 2 * PI * 1e9;
    p.dt = 1e-9 * p.t / p.nt;
    p.C1 = 1. / (2. * p.E_C1);
    p.C2 = 1. / (2. * p.E_C2);
    p.CC = 1. / (2. * p.E_CC);
    rng.reseed(seed);

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