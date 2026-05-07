#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Background image writer. Off-loads PNG (fpng) or raw RGBA8 encoding to a
// single worker thread so the render loop can hand off a captured frame and
// continue with the next pose. The encode runs ~17 ms/frame at 1080p with
// fpng — overlapping it with the next render is the only reason this class
// exists. Format is picked from the output path's extension: `.png` → fpng,
// anything else → raw RGBA8 bytes verbatim.
//
// Failure modes:
//   - failFast=false (default): on encode failure the worker logs and
//     continues. Use failureCount() / firstFailurePath() at end-of-run to
//     summarise. Right for ad-hoc captures (F12) where one bad frame
//     shouldn't kill the GUI.
//   - failFast=true : on the FIRST encode failure the worker drops any
//     queued jobs and refuses further submits. submit() returns false.
//     Right for batch jobs (replay, headless, multi-shot capture) where
//     missing frames quietly corrupt the downstream output (mp4 splice).
class ImageWriter {
public:
    explicit ImageWriter(bool failFast = false, size_t maxQueueDepth = 4);
    ~ImageWriter();

    ImageWriter(const ImageWriter&) = delete;
    ImageWriter& operator=(const ImageWriter&) = delete;

    // Hands ownership of `pixels` (RGBA8, w*h*4 bytes, row-major top-down) to
    // the worker. Blocks if the queue is full so callers naturally throttle
    // when encode can't keep up with capture.
    //
    // Returns false if the writer has shut down (either ~ImageWriter or, in
    // failFast mode, after a prior encode failure). Producers should treat
    // false as a signal to abort the capture loop and propagate the error.
    bool submit(std::vector<unsigned char> pixels,
                uint32_t width,
                uint32_t height,
                std::string path);

    // Drains all pending jobs. Returns when the queue is empty and the worker
    // is idle. Safe to call repeatedly.
    void flush();

    // Cumulative encode failures across the lifetime of this writer.
    size_t failureCount() const { return m_failureCount.load(std::memory_order_relaxed); }

    // Path of the first failed encode (empty if none). Captured under a
    // mutex so the string can be inspected after-the-fact from any thread.
    std::string firstFailurePath() const;

private:
    struct Job {
        std::vector<unsigned char> pixels;
        uint32_t width;
        uint32_t height;
        std::string path;
    };

    void workerLoop();
    static bool encode(const Job& job);

    std::thread             m_worker;
    std::mutex              m_mtx;
    std::condition_variable m_cvNotFull;
    std::condition_variable m_cvNotEmpty;
    std::condition_variable m_cvIdle;
    std::queue<Job>         m_queue;
    size_t                  m_maxQueueDepth;
    bool                    m_busy = false;
    bool                    m_quit = false;

    bool                       m_failFast;
    std::atomic<size_t>        m_failureCount{0};
    mutable std::mutex         m_failureMtx;
    std::string                m_firstFailurePath;
};
