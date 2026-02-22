#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <Eigen/Geometry>

class Logger {
private:
    static Logger* instance;
    std::ofstream logFile;

    Logger(std::string log_file);

public:
    static Logger* getInstance() {
        return getInstance("");
    }
    static Logger* getInstance(std::string log_file);
    void logOneTest(const double err_t, 
                    const double err_r, 
                    const Eigen::MatrixXd transform,
                    const double offset_x, 
                    const double offset_y, 
                    const double offset_yaw);
    void log(const std::string& message);
    void close();

    // Disable copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    ~Logger();
};

#endif // LOGGER_H
