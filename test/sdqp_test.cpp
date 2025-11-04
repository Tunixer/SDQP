#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>
#include <sdqp/sdqp.hpp>

TEST(SdqpTest, Test_Reflection_NaN_Failure)
{
    // Test case based on new ADMM TOPP solver log data
    int m = 18;  // Number of constraints from new log
    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Matrix<double, 3, 1> c;
    Eigen::Matrix<double, 3, 1> x;         // decision variables
    Eigen::Matrix<double, -1, 3> A(m, 3);  // constraint matrix
    Eigen::VectorXd b(m);                  // constraint bound
    // clang-format off
    // Identity matrix Q (from log)
    Q << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    // Linear cost term (from new log)
    c << -1e-6, -0.0127468, -0.0254843;

    // Complete constraint matrix from new log data
    A << 2.983471074380164, 0.0, 0.0,
        -1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        392.5619834710743, -392.5619834710743, 0.0,
        -392.5619834710743, 392.5619834710743, 0.0,
        4631.925404201888, -741.0276842024192, 397.19661157753995,
        3846.8014372597398, 829.2427496818781, -387.9198553646086,
        0.0, 2.9834140746524143, 0.0,
        0.0, -1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 392.57323347107427, -392.5582334710743,
        0.0, -392.57323347107427, 392.5582334710743,
        0.0, 0.0, 2.983243078736165,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, -1.0;

    // Complete constraint bounds from new log data
    b << 1.0,
        -0.000001,
        1000000.0,
        0.000002,
        5.0,
        5.0,
        2.0540307161579134,
        2.0540307161579134,
        1.0,
        -0.000001,
        1000000.0,
        0.004631887246473362,
        5.0+1.93e-5 - 3.16219e-09 - 1.77636e-15,  // a small deviation from the expected value
        5.0+1.93e-5 - 3.16219e-09 - 1.77636e-15,  // a small deviation from the expected value
        1.0,
        0.01167162448514278,
        1000000.0,
        -0.000001;
    // clang-format on
    double minobj = sdqp::sdqp<3>(Q, c, A, b, x);

    std::cout << "Reflection failure test optimal sol: " << x.transpose() << std::endl;
    std::cout << "Reflection failure test optimal obj: " << minobj << std::endl;
    std::cout << "Reflection failure test cons precision: " << (A * x - b).maxCoeff()
              << std::endl;

    // Expected solution from scipy analysis
    Eigen::Vector3d expected_x;
    expected_x << 1.0e-6, 0.004631887246473362, 0.01167162448514278;
    double expected_obj = -0.0002776443219269574;

    // Verify solution is feasible
    EXPECT_TRUE((A * x - b).maxCoeff() < 1e-6);

    // Objective should be negative (minimizing with negative cost coefficients)
    EXPECT_NEAR(minobj, expected_obj, 1e-6);
    EXPECT_NEAR(x(0), expected_x(0), 1e-6);
    EXPECT_NEAR(x(1), expected_x(1), 1e-6);
    EXPECT_NEAR(x(2), expected_x(2), 1e-6);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
