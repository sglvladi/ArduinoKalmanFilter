#ifndef _KalmanFilter_h_
#define _KalmanFilter_h_

#include <BasicLinearAlgebra.h>

using namespace BLA;

template <int NUM_MEAS_DIMS>
class Measurement
{
  public:
    Measurement(){};
    Measurement(BLA::Matrix<NUM_MEAS_DIMS> y_in) : y(y_in){};

    BLA::Matrix<NUM_MEAS_DIMS> y;
};

template <int NUM_STATE_DIMS>
class State
{
  public:
    State(){};
    State(BLA::Matrix<NUM_STATE_DIMS> x_in, BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> P_in)
        : x(x_in), P(P_in){};

    BLA::Matrix<NUM_STATE_DIMS> x;
    BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> P;
};

template <int NUM_STATE_DIMS, int NUM_MEAS_DIMS>
class Model
{
  public:
    Model(){};
    Model(BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> F_k_in, BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> Q_k_in,
          BLA::Matrix<NUM_MEAS_DIMS, NUM_STATE_DIMS> H_k_in, BLA::Matrix<NUM_MEAS_DIMS, NUM_MEAS_DIMS> R_k_in)
        : F_k(F_k_in), Q_k(Q_k_in), H_k(H_k_in), R_k(R_k_in){};
    // Transition model
    BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> F_k;
    BLA::Matrix<NUM_STATE_DIMS, NUM_STATE_DIMS> Q_k;

    // Measurement model
    BLA::Matrix<NUM_MEAS_DIMS, NUM_STATE_DIMS> H_k;
    BLA::Matrix<NUM_MEAS_DIMS, NUM_MEAS_DIMS> R_k;
};

class KalmanFilter
{
  public:
    KalmanFilter(){};

    template <int N, int M>
    State<N> predict(Model<N, M> model, State<N> prior)
    {
        BLA::Matrix<N> x = model.F_k * prior.x;
        BLA::Matrix<N, N> P = model.F_k * prior.P * ~model.F_k + model.Q_k;
        return State<N>(x, P);
    };

    template <int N, int M>
    State<N> update(Model<N, M> model, State<N> prediction, Measurement<M> measurement)
    {
        BLA::Matrix<M> y_pred = measurement.y - model.H_k * prediction.x;
        BLA::Matrix<M, M> S = model.R_k + model.H_k * prediction.P * ~model.H_k;
        BLA::Matrix<N, M> K = prediction.P * ~model.H_k * S.Inverse();

        BLA::Matrix<N> x = prediction.x + K * y_pred;
        BLA::Matrix<N, N> P = prediction.P - K * model.H_k * prediction.P;

        Khist.push_back(K);

        return State<N>(x, P);
    }

    template <int N, int M>
    std::vector<State<N>> smooth(Model<N, M> model, std::vector<State<N>> predictions,
                                 std::vector<State<N>> updates,
                                 std::vector<BLA::Matrix<N, M>> Ks)
    {

        int Nk = updates.size();

        std::vector<State<N>> smoothed;
        smoothed.insert(smoothed.begin(), 1, updates[Nk - 1]);

        for (int i = Nk - 2; i >= 0; i--)
        {
            auto C = updates[i].P * ~model.F_k * predictions[i + 1].P.Inverse();
            auto x_s = updates[i].x + C * (smoothed[i + 1].x - predictions[i + 1].x);
            auto P_s = updates[i].P + C * (smoothed[i + 1].P - predictions[i + 1].P) * ~C;
            smoothed.insert(smoothed.begin(), 1, State<N>(x_s, P_s));
        }
        return smoothed;
    }

    std::vector<BLA::Matrix<4, 2>> Khist;
};

#endif