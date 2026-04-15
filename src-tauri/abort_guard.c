/*
 * abort_guard.c — Recover from C++ exceptions that cross extern "C" boundaries.
 *
 * When sherpa-onnx's prebuilt static libraries are linked into a Rust binary,
 * their C++ exception tables may not survive the link step on macOS ARM64.
 * A C++ exception from ONNX Runtime then reaches std::terminate → abort().
 *
 * This guard installs a SIGABRT handler before calling into sherpa-onnx.
 * If abort() fires, siglongjmp recovers to the setjmp point instead of
 * killing the process.  The caller gets a clean failure return.
 *
 * Only used on Unix (macOS / Linux).  Android doesn't need it because the
 * C++ runtime is loaded as a shared library with working exception tables.
 */

#include <setjmp.h>
#include <signal.h>
#include <stddef.h>

static __thread sigjmp_buf g_buf;
static __thread volatile sig_atomic_t g_active;
static __thread struct sigaction g_old_sa;

static void trap(int sig) {
    (void)sig;
    if (g_active) {
        siglongjmp(g_buf, 1);
    }
    /* Not guarded — restore default and re-raise. */
    signal(SIGABRT, SIG_DFL);
    raise(SIGABRT);
}

/*
 * Begin an abort-guarded region.
 * Returns 0 on first call (normal path).
 * Returns 1 if recovered from SIGABRT (abort path).
 * Caller MUST call sherpa_abort_guard_leave() on BOTH paths.
 */
int sherpa_abort_guard_enter(void) {
    struct sigaction sa;
    sa.sa_handler = trap;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGABRT, &sa, &g_old_sa);
    g_active = 1;
    return sigsetjmp(g_buf, 1);
}

void sherpa_abort_guard_leave(void) {
    g_active = 0;
    sigaction(SIGABRT, &g_old_sa, NULL);
}
