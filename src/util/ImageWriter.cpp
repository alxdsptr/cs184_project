#include "util/ImageWriter.h"
#include "util/Log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <mutex>

#include <fpng.h>

namespace {
std::string lowercaseExt(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return {};
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return ext;
}

bool writeRawRGBA8(const std::string& path,
                   const unsigned char* pixels,
                   size_t bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = std::fwrite(pixels, 1, bytes, f);
    std::fclose(f);
    return written == bytes;
}

void initFpngOnce() {
    static std::once_flag flag;
    std::call_once(flag, [] { fpng::fpng_init(); });
}
}  // namespace

ImageWriter::ImageWriter(bool failFast, size_t maxQueueDepth)
    : m_maxQueueDepth(maxQueueDepth == 0 ? 1 : maxQueueDepth),
      m_failFast(failFast) {
    m_worker = std::thread([this] { workerLoop(); });
}

ImageWriter::~ImageWriter() {
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_quit = true;
    }
    m_cvNotEmpty.notify_all();
    m_cvNotFull.notify_all();
    if (m_worker.joinable()) m_worker.join();
}

bool ImageWriter::submit(std::vector<unsigned char> pixels,
                         uint32_t width,
                         uint32_t height,
                         std::string path) {
    Job job;
    job.pixels = std::move(pixels);
    job.width  = width;
    job.height = height;
    job.path   = std::move(path);

    std::unique_lock<std::mutex> lk(m_mtx);
    m_cvNotFull.wait(lk, [this] {
        return m_quit || m_queue.size() < m_maxQueueDepth;
    });
    if (m_quit) return false;
    m_queue.push(std::move(job));
    m_cvNotEmpty.notify_one();
    return true;
}

void ImageWriter::flush() {
    std::unique_lock<std::mutex> lk(m_mtx);
    m_cvIdle.wait(lk, [this] { return m_queue.empty() && !m_busy; });
}

std::string ImageWriter::firstFailurePath() const {
    std::lock_guard<std::mutex> g(m_failureMtx);
    return m_firstFailurePath;
}

void ImageWriter::workerLoop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cvNotEmpty.wait(lk, [this] { return m_quit || !m_queue.empty(); });
            if (m_quit && m_queue.empty()) return;
            job = std::move(m_queue.front());
            m_queue.pop();
            m_busy = true;
            m_cvNotFull.notify_one();
        }

        const bool ok = encode(job);
        if (!ok) {
            LOG_ERROR("ImageWriter: failed to write %s", job.path.c_str());
            m_failureCount.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> g(m_failureMtx);
                if (m_firstFailurePath.empty()) m_firstFailurePath = job.path;
            }
        }

        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_busy = false;
            // Fail-fast: poison the writer so any further submit returns
            // false and flush() unblocks. Drop pending jobs — if encode is
            // failing for a structural reason (disk full, perms), retrying
            // them just generates more error spam.
            if (!ok && m_failFast) {
                m_quit = true;
                while (!m_queue.empty()) m_queue.pop();
                m_cvNotFull.notify_all();
                m_cvIdle.notify_all();
                return;
            }
            if (m_queue.empty()) m_cvIdle.notify_all();
        }
    }
}

bool ImageWriter::encode(const Job& job) {
    std::filesystem::path p(job.path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    const size_t expected = (size_t)job.width * (size_t)job.height * 4u;
    if (job.pixels.size() != expected) {
        LOG_ERROR("ImageWriter: pixel buffer size mismatch (have %zu, want %zu)",
                  job.pixels.size(), expected);
        return false;
    }

    const std::string ext = lowercaseExt(job.path);
    if (ext == ".png") {
        initFpngOnce();
        return fpng::fpng_encode_image_to_file(
            job.path.c_str(), job.pixels.data(),
            job.width, job.height, /*num_chans=*/4,
            /*flags=*/0);
    }
    return writeRawRGBA8(job.path, job.pixels.data(), job.pixels.size());
}
