/*
Generate one-dimensional transmon phase configurations with local Metropolis
updates.

Inputs: time-slice count Nt, total time t, charging energy E_C, Josephson energy
E_J, decorrelation sweeps, sample count, output path, and optional RNG seed.
Output: binary file with dof and sample-count integers followed by sampled
doubles in Eigen column-major order.
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
struct Index
{
    int x0, x1;
};

// Physical parameters and Monte Carlo controls for the transmon chain.
struct params
{
    int nt, dof;
    double E_C, E_J, t, dt, delta;
    double C;

    int n_decor, n_thermal, n_conf;
};

/*
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
*/

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
// Evaluate the complete periodic transmon action for phase configuration A.
double Action(Eigen::ArrayXd &A, int n, params &p)
{
    double pot, kin, diff;

    kin = 0.0;
    pot = 0.0;

    for (int i=0; i<p.dof; i++)
    {
        diff = A[(i+1)%p.dof] - A[i];

        kin += diff*diff;

        pot += std::cos(A[i]);
    }


    kin = 1./8 * p.C * kin / p.dt;
    pot = -p.E_J * pot * p.dt;

    return kin+pot;
}


// Evaluate the terms in the action that depend on phase component n.
double Action_Local(Eigen::ArrayXd &A, int n, params &p)
{
    /*
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
    */

    double pot, kin;

    kin = 1.0/8.0 * p.C * ( pow( A[(n+1)%p.dof] - A[n], 2) + pow( A[(n-1 + p.dof)%p.dof] - A[n], 2) )/ p.dt;
    pot = -p.E_J * std::cos(A[n]) * p.dt;

    return kin + pot;
}


// Propose and accept/reject a local phase change at component n.
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
    return mc::calibrate(A, p, p.dof, Metropolis, random);
}

// Parse transmon parameters and write a binary configuration file.
int main(int argc, char **argv)
{
    struct params p;
    p.delta = 1;

    unsigned int seed = 42;
    double charging_energy_ghz;
    double josephson_energy_ghz;
    std::string output_path;
    CLI::App app{"Generate periodic transmon phase configurations"};
    app.add_option("--time-slices", p.nt, "Number of time slices")->required();
    app.add_option("--total-time-ns", p.t, "Total simulation time in nanoseconds")->required();
    app.add_option("--charging-energy-ghz", charging_energy_ghz, "Charging energy in GHz")->required();
    app.add_option("--josephson-energy-ghz", josephson_energy_ghz, "Josephson energy in GHz")->required();
    app.add_option("--decorrelation-sweeps", p.n_decor, "Sweeps between samples")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

    p.dof = p.nt;
    p.E_C = 2 * PI * 1e9 * charging_energy_ghz;
    p.E_J = 2 * PI * 1e9 * josephson_energy_ghz;
    p.dt = 1e-9 * p.t / p.nt;
    p.C = 1. / (2. * p.E_C);
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