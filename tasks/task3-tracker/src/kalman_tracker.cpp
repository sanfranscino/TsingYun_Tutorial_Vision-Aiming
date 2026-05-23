#include "kalman_tracker.hpp"

#include <algorithm>
#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

   void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        dt = std::max(0.0, dt);

        position = position + velocity * dt;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        const double q00 = process_noise * dt4 / 4.0;
        const double q01 = process_noise * dt3 / 2.0;
        const double q10 = process_noise * dt3 / 2.0;
        const double q11 = process_noise * dt2;

        const double old_p00 = p00;
        const double old_p01 = p01;
        const double old_p10 = p10;
        const double old_p11 = p11;

        p00 = old_p00 + dt * old_p10 + dt * old_p01 + dt2 * old_p11 + q00;
        p01 = old_p01 + dt * old_p11 + q01;
        p10 = old_p10 + dt * old_p11 + q10;
        p11 = old_p11 + q11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        // 1. 残差：测量值 - 当前预测值
        const double residual = measured_position - position;

        // 2. S = H * P * H^T + R
        // 因为 H = [1, 0]，所以 S = p00 + measurement_noise
        const double s = p00 + measurement_noise;

        // 如果 S 不合理，就不更新
        if (s <= 1e-12)
        {
            return;
        }

        // 3. K = P * H^T / S
        // 因为 H = [1, 0]，所以 K = [p00 / S, p10 / S]^T
        const double k0 = p00 / s;
        const double k1 = p10 / s;

        // 4. 更新状态
        position = position + k0 * residual;
        velocity = velocity + k1 * residual;

        // 5. 更新协方差 P = (I - K * H) * P
        const double old_p00 = p00;
        const double old_p01 = p01;
        const double old_p10 = p10;
        const double old_p11 = p11;

        p00 = (1.0 - k0) * old_p00;
        p01 = (1.0 - k0) * old_p01;
        p10 = old_p10 - k1 * old_p00;
        p11 = old_p11 - k1 * old_p01;
    }
    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        if (!tracking_)
        {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);

            tracking_ = true;

            return stateFromFilters();
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        if (!tracking_)
        {
            return {};
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        return stateFromFilters();
    }
    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
