/*
Generate two-dimensional open-boundary SU(2) configurations using Euler-angle
coordinates and Metropolis updates.

Inputs: coupling g, sample count, output path, optional decorrelation sweeps,
and optional RNG seed; lattice geometry is defined in this source. Output:
binary file with dof and sample-count integers followed by sampled doubles in
Eigen column-major order.
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
// SU(2) model coupling and Monte Carlo run controls.
struct params
{
    int dof, n_decor;
    double g, delta;
    int n_thermal, n_conf;
};

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
// Evaluate the Euler-coordinate action for the current SU(2) angle vector.
double Action(Eigen::ArrayXd &A, params &p)
{
    if (A[1] < 0 || A[1] > PI)
    {
        return pow(10, 10);
    }
    else
    {
        return -4. / pow(p.g, 2) * cos(A[0] / 2.) - log(pow(sin(A[0] / 2.), 2) * sin(A[1]));
    }
}

// Propose a Gaussian change to the angle vector and apply Metropolis acceptance.
Eigen::ArrayXd Metropolis(Eigen::ArrayXd &A, params &p)
{
    Eigen::ArrayXd A_new = A + p.delta * Eigen::ArrayXd::NullaryExpr(p.dof, [&]()
                                                                     { return random.proposal(); });

    double dS = Action(A_new, p) - Action(A, p);

    if (random.accept(dS))
    {
        return A_new;
    }
    else
    {
        return A;
    }
}

// Collect n_conf states separated by decorrelation sweeps.
Eigen::MatrixXd Sweep(Eigen::ArrayXd &A, params &p)
{
    return mc::sweep(A, p, p.n_decor,
                     [](Eigen::ArrayXd &state, int, params &parameters)
                     { return Metropolis(state, parameters); });
}

// Evolve the configuration for the configured thermalization interval.
Eigen::ArrayXd Thermalization(Eigen::ArrayXd &A, params &p)
{
    return mc::thermalize(A, p, p.dof, p.n_thermal,
                          [](Eigen::ArrayXd &state, int, params &parameters)
                          { return Metropolis(state, parameters); });
}

// Tune the proposal width until the measured acceptance fraction is in range.
Eigen::ArrayXd Calibrate(Eigen::ArrayXd &A, params &p)
{
    return mc::calibrate(A, p, p.dof,
                         [](Eigen::ArrayXd &state, int, params &parameters)
                         { return Metropolis(state, parameters); },
                         random);
}

// Parse coupling/sample/output arguments and write a binary sample file.
int main(int argc, char **argv)
{
    struct params p;
    p.delta = 1;
    p.dof = pow(2, 2) - 1;
    p.n_decor = 1;

    unsigned int seed = 42;
    std::string output_path;
    CLI::App app{"Generate open-boundary SU(2) configurations in Euler coordinates"};
    app.add_option("--coupling", p.g, "SU(2) coupling")->required();
    app.add_option("--samples", p.n_conf, "Number of configurations to generate")->required();
    app.add_option("--decorrelation-sweeps", p.n_decor, "Sweeps between samples")
        ->default_val(1);
    mc::add_output_options(app, output_path, seed);
    mc::add_thermalization_option(app, p.n_thermal);
    CLI11_PARSE(app, argc, argv);

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