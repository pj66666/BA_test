//
// Created by gaoxiang19 on 11/10/18.
//

#include "backend/loss_function.h"

namespace myslam {
namespace backend {
/**
 * Huber loss
 *
 * Huber(e) = e^2                      if e <= delta
 * huber(e) = delta*(2*e - delta)      if e > delta
 */
double HuberLoss::Compute(double error) const {
    if (error <= delta_) {
        return error * error;
    } else {
        return delta_ * (2 * error - delta_);
    }
}

}
}
