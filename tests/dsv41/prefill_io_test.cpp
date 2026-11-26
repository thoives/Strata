// Chunked prefill I/O with real queue workers, synthetic tier stripes, and no model or GPU.
#include "ds4_prefill_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strata::ds4;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Fail fast: unwinding past a queue whose worker is still gated would block in its destructor.
#define CHECK(expr)                                                                    \
    do {                                                                               \
        if (!(expr)) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s failed\n", __func__, __LINE__, #expr);      \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (false)

class Gate {
public:
    void hold() {
        std::unique_lock<std::mutex> lock(mu_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return open_; });
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mu_);
        CHECK(cv_.wait_for(lock, 5s, [&] { return entered_; }));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mu_);
        open_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool entered_ = false, open_ = false;
};

// Install before creating queues and clear only after all their workers have joined.
class ReadHook {
public:
    explicit ReadHook(std::function<bool(const IoQueue::Job&)> fn) : fn_(std::move(fn)) {
        active_ = this;
        IoQueue::read_hook() = [](const IoQueue::Job& job) { return active_->fn_(job); };
    }
    ~ReadHook() { IoQueue::read_hook() = nullptr; active_ = nullptr; }

private:
    static inline ReadHook* active_ = nullptr;
    std::function<bool(const IoQueue::Job&)> fn_;
};

struct Tier {
    struct StripeRef { int dev; const uint8_t* src; uint64_t blob_off, len; };
    std::array<IoQueue, 2> queues{IoQueue(1), IoQueue(1)};
    std::array<std::vector<StripeRef>, 2> stripes;

    size_t device_count() const { return queues.size(); }
    IoQueue* queue(int dev) { return &queues[dev]; }
    bool stripe_refs(int layer, int expert, std::vector<StripeRef>& out) const {
        if (layer != 0 || expert < 0 || (size_t) expert >= stripes.size()) return false;
        out = stripes[expert];
        return !out.empty();
    }
};

template <size_t N>
std::array<uint8_t, N> pattern(unsigned salt) {
    std::array<uint8_t, N> bytes{};
    for (size_t i = 0; i < N; ++i) bytes[i] = (uint8_t) (1 + (37 * i + (i >> 3) + salt) % 251);
    return bytes;
}

struct TempFile {
    // An unaligned read crosses the direct-alignment boundary even on the buffered path.
    static constexpr uint64_t off = os::kDirectAlign - 3;
    const std::array<uint8_t, 2 * os::kDirectAlign> bytes = pattern<2 * os::kDirectAlign>(100);
    fs::path dir;
    os::fd_t fd;

    TempFile() {
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = getpid();
#endif
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        dir = fs::temp_directory_path() / ("strata-prefill-io-" + std::to_string(pid) + "-" + std::to_string(stamp));
        CHECK(fs::create_directory(dir));
        const auto path = dir / "weights.bin";
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        out.close();
        CHECK(out);
        const auto utf8 = path.u8string();
        fd = os::open_par(std::string(utf8.begin(), utf8.end()));
        CHECK(fd != os::kBadFd);
    }
    ~TempFile() {
        os::close_fd(fd);
        fs::remove_all(dir);
    }
};

// Use the prefill wakeup path with a deadline, so a broken overlap test fails at the stalled read.
void drain(const PrefillReads& reads, const PrefillRead& read) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    for (;;) {
        const auto seen = reads.epoch();
        if (read.left.load(std::memory_order_acquire) == 0) return;
        CHECK(std::chrono::steady_clock::now() < deadline);
        reads.wait_change(seen, 50'000);
    }
}

void parallel_mixed_reads(const TempFile& file) {
    const auto a_src = pattern<24>(10), b_src = pattern<24>(20);
    std::array<uint8_t, 26> a_dst{}, keep{}, expected_dst{}, expected_keep{};
    std::array<uint8_t, 8> b_dst{};
    std::array<uint8_t, os::kDirectAlign + 10> file_dst{}, expected_file_dst{};
    std::array<uint8_t, 9> file_keep{};
    PrefillReads reads;
    struct Item : PrefillRead { int stage; };
    Item items[3];
    for (int i = 0; i < 3; ++i) items[i].stage = i;
    const auto& a = items[0];
    const auto& b = items[1];
    const auto& f = items[2];
    Gate a_stripe;
    ReadHook hook([&](const IoQueue::Job& job) {
        if (job.mem == a_src.data() + 5) a_stripe.hold();
        return true;
    });
    Tier tier;
    IoQueue file_queue(1);
    tier.stripes[0] = {{0, a_src.data(), 0, 8}, {1, a_src.data() + 8, 8, 8}, {1, a_src.data() + 16, 16, 8}};
    tier.stripes[1] = {{1, b_src.data(), 0, 8}};
    // The first range starts and ends inside stripes and spans all three. The second skips a gap and ends at the blob end.
    const PrefillRange a_ranges[] = {{5, 14, a_dst.data() + 2, keep.data() + 1}, {21, 3, a_dst.data() + 20}};
    const PrefillRange b_range{0, b_dst.size(), b_dst.data()};
    IoQueue::Job job;
    job.fd = file.fd;
    job.off = TempFile::off;
    job.len = file_keep.size();
    job.dst = file_dst.data();
    job.keep = file_keep.data();
    size_t first = 0;
    submit_prefill_window(first, 3, 3, [&](size_t i) -> Item& { return items[i]; }, [&](size_t i, Item& it) {
        const PrefillRange* ranges = i == 0 ? a_ranges : &b_range;
        if (!reads.submit_tier(tier, 0, (int) i, ranges, i == 0 ? 2 : 1, it, false))
            reads.submit_file(file_queue, &job, 1, it);
        return true;
    });
    CHECK(first == 3);
    a_stripe.wait_entered();
    CHECK(!a.ready());
    drain(reads, b);
    CHECK(b.ready() && std::equal(b_dst.begin(), b_dst.end(), b_src.begin()));
    // B was behind A's device-1 work; it started and finished without releasing A's device-0 stripe.
    CHECK(a.left.load() == 1 && !a.ready());
    drain(reads, f);
    const auto* file_src = file.bytes.data() + job.off;
    std::copy_n(file_src, job.len, expected_file_dst.data() + job.off % os::kDirectAlign);
    CHECK(f.ready() && !f.bad && file_dst == expected_file_dst);
    CHECK(std::equal(file_keep.begin(), file_keep.end(), file_src));
    CHECK(!a.ready());

    a_stripe.release();
    drain(reads, a);
    std::copy_n(a_src.data() + 5, 14, expected_dst.data() + 2);
    std::copy_n(a_src.data() + 21, 3, expected_dst.data() + 20);
    std::copy_n(a_src.data() + 5, 14, expected_keep.data() + 1);
    CHECK(a.ready() && a_dst == expected_dst && keep == expected_keep);
    CHECK(reads.tier_reads == 2 && reads.tier_bytes == 25 && reads.tier_retries == 0);
}

void staging_window() {
    struct Item { int stage; bool submitted = false; };
    Item items[] = {{0}, {-1}, {1}, {2}, {0}, {1}, {-1}, {2}, {0}};
    constexpr int free = -1, prev_dma = -2;
    std::array<int, 3> owner{prev_dma, free, free};
    size_t first = 0;
    std::vector<size_t> tried;
    auto pump = [&] {
        tried.clear();
        submit_prefill_window(first, std::size(items), owner.size(), [&](size_t i) -> Item& { return items[i]; },
                              [&](size_t i, Item& it) {
            tried.push_back(i);
            if (it.stage < 0) return true;
            if (owner[it.stage] != free) return false;
            owner[it.stage] = (int) i;
            return true;
        });
    };
    // The blocked first claimant must not stop slots 1 and 2. Unstaged items don't count against capacity.
    pump();
    CHECK((first == 0 && tried == std::vector<size_t>{0, 1, 2, 3}));
    CHECK((owner == std::array<int, 3>{prev_dma, 2, 3} && items[1].submitted));
    owner[0] = free;
    pump();
    CHECK(first == 4 && tried == std::vector<size_t>{0});
    CHECK(owner[0] == 0);

    // The ring wraps. A later DMA frees its slot while the earlier claimant's slot is still busy.
    owner[1] = owner[2] = free;
    pump();
    CHECK((first == 4 && tried == std::vector<size_t>{4, 5, 6, 7}));
    CHECK((owner == std::array<int, 3>{0, 5, 7} && items[6].submitted));
    owner[0] = free;
    pump();
    CHECK(first == 8 && tried == std::vector<size_t>{4});
    pump();
    CHECK(first == 8 && tried == std::vector<size_t>{8} && owner[0] == 4);
    owner[0] = free;
    pump();
    CHECK(first == std::size(items) && tried == std::vector<size_t>{8} && owner[0] == 8);
}

void duplicate_routed_picks() {
    const std::vector<int> host_list{1, 2, 3, 4, 5};
    const int vs[] = {0, 3, 1, 2, 4, 5}; // expert 0 is resident; expert 2 has already been speculated
    int bounds[std::size(vs) + 1]{};
    for (int expert : {0, 1, 1, 2, 4, 1, 4, 5, 2, 5, 0, 1}) ++bounds[vs[expert] + 1];
    std::partial_sum(std::begin(bounds), std::end(bounds), bounds);
    std::vector<std::pair<int, int>> routed;
    prefill_routed(host_list, vs, 2, bounds, [&](int e, int v) { routed.emplace_back(e, v); });
    CHECK((routed == std::vector<std::pair<int, int>>{{1, 3}, {4, 4}, {5, 5}}));
}

void drain_and_retry(const TempFile& file) {
    const auto src = pattern<16>(30);
    constexpr size_t payload = 16;
    constexpr uint64_t file_off = os::kDirectAlign + 3;
    std::array<uint8_t, payload + 4> dst{}, keep{}, partial{}, expected{};
    uint8_t marker = 0;
    PrefillReads reads;
    PrefillRead read, fence;
    Gate sibling;
    ReadHook hook([&](const IoQueue::Job& job) {
        if (job.mem == src.data() + 8) sibling.hold();
        return !job.tier_stripe || job.mem != src.data(); // report failure without writing the first stripe
    });
    Tier tier;
    IoQueue file_queue(1);
    tier.stripes[0] = {{0, src.data(), 0, 8}, {1, src.data() + 8, 8, 8}};
    const PrefillRange range{0, payload, dst.data(), keep.data()};
    CHECK(reads.submit_tier(tier, 0, 0, &range, 1, read, false));
    sibling.wait_entered();
    // A marker behind the failed job on device 0 makes the partial failure observable while device 1 is still held.
    IoQueue::Job mark;
    mark.mem = src.data();
    mark.dst = &marker;
    mark.len = 1;
    reads.submit_file(*tier.queue(0), &mark, 1, fence);
    drain(reads, fence);
    CHECK(marker == src[0] && read.bad && read.left.load() == 1 && !read.ready());

    int retries = 0;
    auto retry = [&] {
        ++retries;
        IoQueue::Job job;
        job.fd = file.fd;
        job.off = file_off;
        job.len = payload;
        job.dst = dst.data();
        job.keep = keep.data();
        job.exact = true; // unlike the staged file read above, even this unaligned read lands at dst
        reads.submit_file(file_queue, &job, 1, read);
    };
    CHECK(!reads.retry_file(read, retry) && retries == 0);
    sibling.release();
    reads.wait(read);
    CHECK(read.left.load() == 0);
    std::copy_n(src.data() + 8, 8, partial.data() + 8);
    CHECK(!read.ready() && read.bad && dst == partial && keep == partial);
    CHECK(reads.retry_file(read, retry));
    drain(reads, read);
    std::copy_n(file.bytes.data() + file_off, payload, expected.data());
    CHECK(read.ready() && !read.bad && !read.tier && dst == expected && keep == expected);
    CHECK(!reads.retry_file(read, retry) && retries == 1);
    CHECK(reads.tier_reads == 1 && reads.tier_bytes == payload && reads.tier_retries == 1);
}

void demand_before_speculation() {
    const auto src = pattern<4>(70);
    std::array<uint8_t, 4> dst{};
    std::vector<int> start_order;
    Gate busy;
    PrefillReads reads;
    PrefillRead read[4];
    ReadHook hook([&](const IoQueue::Job& job) {
        const int i = (int) (job.mem - src.data());
        start_order.push_back(i);
        if (i == 0) busy.hold();
        return true;
    });
    IoQueue queue(1);
    auto submit = [&](int i, bool low) {
        IoQueue::Job job;
        job.mem = src.data() + i;
        job.dst = dst.data() + i;
        job.len = 1;
        reads.submit_file(queue, &job, 1, read[i], low);
    };
    submit(0, false);
    busy.wait_entered();
    submit(1, true);
    submit(2, true);
    submit(3, false);
    queue.promote(read[2].left);
    busy.release();
    for (const auto& r : read) reads.wait(r);
    CHECK((start_order == std::vector<int>{0, 3, 2, 1} && dst == src));
}

void cancel_running_and_queued() {
    const auto src = pattern<24>(40);
    std::array<uint8_t, 16> dst{}, keep{}, partial{};
    std::array<uint8_t, 8> queued_dst{}, queued_keep{}, untouched{};
    PrefillReads reads;
    PrefillRead running, queued;
    Gate stripe;
    ReadHook hook([&](const IoQueue::Job& job) {
        if (job.mem == src.data()) stripe.hold();
        return true;
    });
    Tier tier;
    tier.stripes[0] = {{0, src.data(), 0, 8}, {0, src.data() + 8, 8, 8}};
    tier.stripes[1] = {{0, src.data() + 16, 0, 8}};
    const PrefillRange a{0, dst.size(), dst.data(), keep.data()};
    const PrefillRange b{0, queued_dst.size(), queued_dst.data(), queued_keep.data()};
    CHECK(reads.submit_tier(tier, 0, 0, &a, 1, running, false));
    stripe.wait_entered(); // only worker on device 0: A's second stripe and all of B have not started
    CHECK(reads.submit_tier(tier, 0, 1, &b, 1, queued, true));
    CHECK(running.left.load() == 2 && queued.left.load() == 1);
    reads.cancel();
    stripe.release();
    drain(reads, running);
    drain(reads, queued);
    std::copy_n(src.data(), 8, partial.data());
    CHECK(running.bad && !running.ready() && dst == partial && keep == partial);
    CHECK(queued.bad && !queued.ready() && queued_dst == untouched && queued_keep == untouched);
    CHECK(!reads.retry_file(running, [] { CHECK(false); }) && reads.tier_retries == 0);
}
}  // namespace

int main() {
    const TempFile file;
    parallel_mixed_reads(file);
    staging_window();
    duplicate_routed_picks();
    drain_and_retry(file);
    demand_before_speculation();
    cancel_running_and_queued();
    std::puts("ds4_prefill_io_test: all passed");
}
