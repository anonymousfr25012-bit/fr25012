#include <boomer2_tools/Logger.hh>


// Example usage:
int main() {
    Logger* logger = Logger::getInstance();

    // Log some messages
    logger->log("Test result: Passed");
    logger->log("Test result: Failed");

    double et = 0.1;
    double er = 0.2;
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(4, 4);
    logger->logOneTest(et, er, matrix,0,0,0  );

    et = 1;
    er = 2;

    logger->logOneTest(et, er, matrix,0,0,0  );

    // Close the logger when done
    logger->close();

    return 0;
}
