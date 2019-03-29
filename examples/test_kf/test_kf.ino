#include <ArduinoKalmanFilter.h>
#include <BasicLinearAlgebra.h>

using namespace BLA;

BLA::Matrix<4> x = {1, 0, 1, 1};
BLA::Matrix<4, 4> P = {0.2, 0, 0, 0,
                       0, 0.2, 0, 0,
                       0, 0, 0.2, 0,
                       0, 0, 0, 0.2};
State<4> prior = State<4>(x, P);

BLA::Matrix<4, 4> F_k = {1, 1, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 1,
                         0, 0, 0, 1};
BLA::Matrix<4, 4> Q_k = {1 / 3, 1 / 2, 0, 0,
                         1 / 2, 1, 0, 0,
                         0, 0, 1 / 3, 1 / 2,
                         0, 0, 1 / 2, 1};
BLA::Matrix<2, 4> H_k = {1, 0, 0, 0,
                         0, 0, 1, 0};
BLA::Matrix<2, 2> R_k = {0.1, 0,
                         0, 0.1};
Model<4, 2> model = Model<4, 2>(F_k, Q_k, H_k, R_k);

BLA::Matrix<2> y = {1.5, 1.5};
Measurement<2> meas = Measurement<2>(y);

ArduinoKalmanFilter kf;

void setup()
{
  model.Q_k = model.Q_k * 0.0001;
  Serial.begin(115200);

  Serial << "x_prior: " << prior.x << ";\nP_prior: " << prior.P << "\n";
  auto prediction = kf.predict(model, prior);
  Serial << "x_pred: " << prediction.x << ";\nP_pred: " << prediction.P << "\n";
  auto posterior = kf.update(model, prediction, meas);
  Serial << "x_new: " << posterior.x << ";\nP_new: " << posterior.P << "\n";
}
void loop()
{
}
