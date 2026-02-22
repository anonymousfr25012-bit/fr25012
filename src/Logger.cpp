#include <boomer2_tools/Logger.hh>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>  // Include this header for std::setprecision

Logger* Logger::instance = nullptr;

Logger::Logger(std::string log_file) {
    // Get the current date and time
    std::time_t rawTime;
    std::time(&rawTime);

   
    // Format the timestamp as a string
    struct std::tm* timeInfo = std::localtime(&rawTime);
    
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeInfo);
    std::string timestampStr(buffer);

    // Create the log filename with the timestamp
    std::string logFilename = "log_" + timestampStr + "_" + log_file + ".txt";

    logFile.open(logFilename, std::ios::app);

    return;
    
}

Logger* Logger::getInstance(std::string log_file = "") {
    if (!instance) {
        instance = new Logger(log_file);
    }
    return instance;
}

void Logger::logOneTest(const double err_t, 
                        const double err_r, 
                        const Eigen::MatrixXd transform, 
                        const double offset_x, 
                        const double offset_y, 
                        const double offset_yaw) {
    logFile << "----------------" << std::endl;
    
    logFile << "Offsets (dx, dy, dz): " << std::setprecision(4) 
            << offset_x << ", "
            << offset_y << ", "
            << offset_yaw << std::endl;

    logFile << "Transformation:" << std::endl 
            << transform << std::endl;

    logFile << "Translation error: " << std::setprecision(4) << err_t << "; "
            << "Rotation error: " << err_r << ";" << std::endl;
}
void Logger::log(const std::string& message) {
    logFile << message << std::endl;
}

void Logger::close() {
    logFile.close();
    delete instance;
    instance = nullptr;
}

Logger::~Logger() {
    logFile.close();
}
