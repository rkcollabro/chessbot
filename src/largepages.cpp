#include "largepages.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "advapi32.lib")

namespace {
// Large pages need SeLockMemoryPrivilege granted to the account and then enabled in this process.
bool enableLockMemoryPrivilege() {
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;
    CloseHandle(token);
    return ok;
}
}  // namespace

void* allocLargePages(size_t bytes, bool& large) {
    large = false;
    const size_t page = GetLargePageMinimum();
    if (page && enableLockMemoryPrivilege()) {
        void* p = VirtualAlloc(nullptr, (bytes + page - 1) / page * page, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            large = true;
            return p;
        }
    }
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);  // zeroed by the OS
}

void freeLargePages(void* p) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

#else
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

void* allocLargePages(size_t bytes, bool& large) {
    large = false;
    constexpr size_t PAGE = size_t(2) << 20;
    const size_t size = (bytes + PAGE - 1) / PAGE * PAGE;
    void* p = std::aligned_alloc(PAGE, size);
    if (!p) return nullptr;
#ifdef __linux__
    large = madvise(p, size, MADV_HUGEPAGE) == 0;  // a request: the kernel backs it with huge pages when it can
#endif
    std::memset(p, 0, size);
    return p;
}

void freeLargePages(void* p) { std::free(p); }
#endif
