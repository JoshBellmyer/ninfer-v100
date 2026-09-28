// Minimal process/console portability shims shared by host-side code that needs a per-process
// identity or stderr terminal properties. Everything else stays on the standard library.
#pragma once

#include <cstdint>

#if defined(_WIN32)
#    include <process.h>
#    include <windows.h>
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

namespace ninfer::platform {

// Identity of the current process, used to make temporary file names unique per run.
[[nodiscard]] inline std::uint64_t process_id() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

// True when stderr is attached to an interactive terminal (POSIX TTY or Windows console).
[[nodiscard]] inline bool stderr_is_interactive() noexcept {
#if defined(_WIN32)
    DWORD mode = 0;
    return ::GetConsoleMode(::GetStdHandle(STD_ERROR_HANDLE), &mode) != 0;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

// Width in columns of the console attached to stderr, or 0 when stderr is not a terminal.
[[nodiscard]] inline int stderr_columns() noexcept {
#if defined(_WIN32)
    const HANDLE error = ::GetStdHandle(STD_ERROR_HANDLE);
    const HANDLE output = ::GetStdHandle(STD_OUTPUT_HANDLE);
    for (const HANDLE handle : {error, output}) {
        CONSOLE_SCREEN_BUFFER_INFO info {};
        if (::GetConsoleScreenBufferInfo(handle, &info) && info.dwSize.X > 0) {
            return static_cast<int>(info.dwSize.X);
        }
    }
    return 0;
#else
    winsize size {};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) {
        return static_cast<int>(size.ws_col);
    }
    return 0;
#endif
}

} // namespace ninfer::platform
