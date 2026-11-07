// tests/core/expert_tier_lock_test.cpp - the tier's protection across processes and its data checks (§5.4), on /tmp only.
//   - a regular-file tier is held by an exclusive lock from its open until its close: a second process cannot open it, and
//     the lock goes when the holder exits or closes, also after an error;
//   - "blank" means every byte the fill may write (the mib) is zero: a sparse file with data past its first 128 KiB is refused;
//   - two names of one device are refused before anything is opened, by what they are.
// Host only: no CUDA, no download. POSIX (fork); the Windows lock is not built here.
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_tier_dev.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond, what)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "expert_tier_lock_test: FAILED: " << what << "\n";      \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

static TierIdentity id_for_test() {
    TierIdentity id;
    id.pack_id = "sha:0000000000000000000000000000000000000000000000000000000000000000";
    id.profile_id = "sha:1111111111111111111111111111111111111111111111111111111111111111";
    id.n_layers = 1;
    id.n_expert = 4;
    id.layer_blob_bytes = {4096};
    return id;
}

static fs::path make_device(const fs::path& dir, const std::string& name, uint64_t mib) {
    const fs::path p = dir / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.close();
    fs::resize_file(p, mib << 20);
    return p;
}

static void write_at(const fs::path& p, uint64_t off, const std::string& text) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp((std::streamoff) off);
    f << text;
}

static void test_sparse_data_past_the_prefix_is_refused(const fs::path& dir) {
    const fs::path dev = make_device(dir, "sparse.img", 8);
    write_at(dev, 3ull << 20, "someone's data");              // 3 MiB in: the first 128 KiB are zeros
    ExpertTierSet s;
    std::string err;
    bool reused = true;
    CHECK(!s.open({{dev.string(), 8, 0, false}}, id_for_test(), err, reused), "sparse data inside the mib was taken as blank");
    CHECK(err.find("not a Strata expert tier") != std::string::npos, "the refusal does not name the data: " + err);
    std::ifstream in(dev, std::ios::binary);
    std::string back(14, '\0');
    in.seekg(3ll << 20);
    in.read(&back[0], 14);
    CHECK(back == "someone's data", "a refused device was written");
}

static void test_the_lock_is_held_by_another_process(const fs::path& dir) {
    const fs::path dev = make_device(dir, "locked.img", 8);
    int ready[2];
    CHECK(::pipe(ready) == 0, "pipe");
    const pid_t child = ::fork();
    CHECK(child >= 0, "fork");
    if (child == 0) {                                          // the other process: holds the tier until told to stop
        ::close(ready[0]);
        ExpertTierSet s;
        std::string err;
        bool reused = false;
        const bool ok = s.open({{dev.string(), 8, 0, false}}, id_for_test(), err, reused);
        const char c = ok ? 'y' : 'n';
        (void) !::write(ready[1], &c, 1);
        ::close(ready[1]);
        for (;;) ::pause();                                    // holds the tier until the parent stops it

    }
    ::close(ready[1]);
    char c = 0;
    CHECK(::read(ready[0], &c, 1) == 1 && c == 'y', "the other process could not open the tier");
    ::close(ready[0]);
    {   // a second process (here: this one, another open file) is refused while the other holds it
        ExpertTierSet s;
        std::string err;
        bool reused = false;
        CHECK(!s.open({{dev.string(), 8, 0, false}}, id_for_test(), err, reused), "a tier in use by another process was opened");
        CHECK(err.find("another process holds") != std::string::npos, "the refusal does not say who holds it: " + err);
    }
    // the other process ends: its lock goes with it, and the tier opens again
    ::kill(child, SIGKILL);
    int status = 0;
    ::waitpid(child, &status, 0);
    ExpertTierSet again;
    std::string err;
    bool reused = false;
    CHECK(again.open({{dev.string(), 8, 0, false}}, id_for_test(), err, reused), "the lock was not released when the holder ended: " + err);
    again.close();
}

static void test_the_lock_is_released_after_an_error(const fs::path& dir) {
    const fs::path small = make_device(dir, "small.img", 2);   // smaller than mib=8: refused at open, after the lock was taken
    {
        ExpertTierSet s;
        std::string err;
        bool reused = false;
        CHECK(!s.open({{small.string(), 8, 0, false}}, id_for_test(), err, reused), "a too-small device was opened");
    }
    const fs::path dev = make_device(dir, "after.img", 8);
    ExpertTierSet s;
    std::string err;
    bool reused = false;
    CHECK(s.open({{dev.string(), 8, 0, false}}, id_for_test(), err, reused), "a device refused earlier was left locked: " + err);
    s.close();
}

static void test_data_scan_reads_only_what_is_there(const fs::path& dir) {
    const fs::path dev = make_device(dir, "scan.img", 16);     // all zeros, sparse
    int fd = ::open(dev.c_str(), O_RDWR);
    CHECK(fd >= 0, "open for the scan");
    CHECK(!tier_fd_has_data(fd, 16ull << 20), "an all-zero sparse file has data");
    CHECK(!tier_fd_has_data(fd, 0), "an empty range has data");
    (void) !::pwrite(fd, "z", 1, (off_t) (9ull << 20));
    CHECK(tier_fd_has_data(fd, 16ull << 20), "data at 9 MiB not found");
    CHECK(!tier_fd_has_data(fd, 9ull << 20), "data past the range was found inside it");
    ::close(fd);
}

static void test_two_names_are_refused_before_the_lock(const fs::path& dir) {
    const fs::path dev = make_device(dir, "a.img", 8);
    const fs::path link = dir / "a.link";
    fs::create_symlink(dev, link);
    ExpertTierSet s;
    std::string err;
    bool reused = false;
    CHECK(!s.open({{dev.string(), 8, 0, false}, {link.string(), 8, 0, false}}, id_for_test(), err, reused), "two names opened");
    CHECK(err.find("same device") != std::string::npos, "the alias is not named as the same device: " + err);
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("expert-tier-lock-" + std::to_string((long long) ::getpid()));
    fs::create_directories(dir);
    test_sparse_data_past_the_prefix_is_refused(dir);
    test_the_lock_is_held_by_another_process(dir);
    test_the_lock_is_released_after_an_error(dir);
    test_data_scan_reads_only_what_is_there(dir);
    test_two_names_are_refused_before_the_lock(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (g_failures != 0) {
        std::cerr << "expert_tier_lock_test: " << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "expert_tier_lock_test: PASS\n";
    return 0;
}
