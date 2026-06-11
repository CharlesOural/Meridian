#include "lio/scan_registration.hpp"

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"

TEST(ScanRegistration, Constructs) {
  meridian::lio::ScanRegistration reg{meridian::FrontendLio{}};
  (void)reg;
  meridian::lio::RegistrationResult result;
  EXPECT_EQ(result.n_corr, 0);
  EXPECT_FALSE(result.converged);
}
