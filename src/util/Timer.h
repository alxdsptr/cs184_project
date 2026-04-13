#pragma once
#include <chrono>

class CpuTimer {
public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop()  { m_end   = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }
private:
    std::chrono::high_resolution_clock::time_point m_start, m_end;
};
