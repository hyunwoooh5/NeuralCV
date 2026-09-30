#ifndef NEURALCV_MONTE_CARLO_HPP
#define NEURALCV_MONTE_CARLO_HPP

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <random>
#include <string>

namespace mc
{
inline void add_output_options(CLI::App &app, std::string &output_path, unsigned int &seed)
{
    app.add_option("--output,-o", output_path, "Path for the binary configuration output")
        ->required();
    app.add_option("--seed", seed, "Random-number generator seed")
        ->default_val(42);
}

inline void add_thermalization_option(CLI::App &app, int &thermalization_sweeps)
{
    thermalization_sweeps = 10000;
    app.add_option("--thermalization-sweeps", thermalization_sweeps,
                   "Metropolis sweeps before sampling")
        ->default_val(10000);
}

class Random
{
public:
    Random() : engine_(42), accepted_(0) {}
    explicit Random(unsigned int seed) : engine_(seed), accepted_(0) {}

    void reseed(unsigned int seed)
    {
        engine_.seed(seed);
        proposal_distribution_.reset();
        acceptance_distribution_.reset();
        accepted_ = 0;
    }

    double proposal()
    {
        return proposal_distribution_(engine_);
    }

    bool accept(double delta_action)
    {
        if (std::exp(-delta_action) >= acceptance_distribution_(engine_))
        {
            ++accepted_;
            return true;
        }
        return false;
    }

    void reset_acceptance()
    {
        accepted_ = 0;
    }

    int accepted() const
    {
        return accepted_;
    }

private:
    std::mt19937 engine_;
    std::uniform_real_distribution<double> proposal_distribution_{-1.0, 1.0};
    std::uniform_real_distribution<double> acceptance_distribution_{0.0, 1.0};
    int accepted_;
};

template <typename State, typename Params, typename Update>
Eigen::MatrixXd sweep(State &state, Params &params, int decorrelation_sweeps, Update update)
{
    Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(params.dof, params.n_conf);
    for (int sample = 0; sample < params.n_conf; ++sample)
    {
        for (int sweep = 0; sweep < decorrelation_sweeps; ++sweep)
        {
            for (int site = 0; site < params.dof; ++site)
            {
                state = update(state, site, params);
            }
        }
        samples.col(sample) = state;
    }
    return samples;
}

template <typename State, typename Params, typename Update>
State thermalize(State &state, Params &params, int dof, int thermalization_sweeps, Update update)
{
    for (int sweep = 0; sweep < thermalization_sweeps; ++sweep)
    {
        for (int site = 0; site < dof; ++site)
        {
            state = update(state, site, params);
        }
    }
    return state;
}

template <typename State, typename Params, typename Update>
State calibrate(State &state, Params &params, int dof, Update update, Random &random)
{
    double ratio = 0.0;
    while (ratio <= 0.3 || ratio >= 0.55)
    {
        random.reset_acceptance();
        for (int sweep = 0; sweep < 10; ++sweep)
        {
            for (int site = 0; site < dof; ++site)
            {
                state = update(state, site, params);
            }
        }
        ratio = static_cast<double>(random.accepted()) / (dof * 10);
        if (ratio >= 0.55)
        {
            params.delta *= 1.02;
        }
        else if (ratio <= 0.3)
        {
            params.delta *= 0.98;
        }
    }
    return state;
}

inline bool write_samples(const std::string &path, int dof, int n_conf, const Eigen::MatrixXd &samples)
{
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output)
    {
        return false;
    }

    output.write(reinterpret_cast<const char *>(&dof), sizeof(int));
    output.write(reinterpret_cast<const char *>(&n_conf), sizeof(int));
    output.write(reinterpret_cast<const char *>(samples.data()), samples.size() * sizeof(double));
    return static_cast<bool>(output);
}
} // namespace mc

#endif