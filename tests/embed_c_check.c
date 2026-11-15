/* The embed headers compile as C, newest first (abi_c_check.c does the same for
 * the SDK headers), and the version macros are integer constant expressions
 * that work in #if. caliper_embed_tests calls the function below from C++ to
 * prove that caliper_embed_api_version has C linkage. */
#include <caliper/live_views.h>
#include <caliper/live.h>
#include <caliper/embed.h>

#if CALIPER_EMBED_VERSION != CALIPER_EMBED_MAKE_VERSION(CALIPER_EMBED_API_VERSION, \
        CALIPER_EMBED_VERSION_MINOR, CALIPER_EMBED_VERSION_PATCH)
#error "CALIPER_EMBED_VERSION does not encode its three parts"
#endif
#if (CALIPER_EMBED_VERSION >> 16) != CALIPER_EMBED_API_VERSION
#error "the major of CALIPER_EMBED_VERSION is not CALIPER_EMBED_API_VERSION"
#endif

uint32_t caliper_embed_c_check_version(void) { return caliper_embed_api_version(); }
