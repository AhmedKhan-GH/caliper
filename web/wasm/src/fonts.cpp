// fonts.cpp: the native app's two faces, embedded in the module (a browser
// has no file system to read them from): IBM Plex Sans Condensed for labels and
// Source Code Pro for data, both under the SIL Open Font License 1.1, the
// unchanged files of the design token package (their licences are beside them
// there). The build passes --embed-dir=<the package font folder>.
#include "fonts.h"

namespace {
const unsigned char kUi[] = {
#embed "IBMPlexSansCondensed-Regular.ttf"
};
const unsigned char kMono[] = {
#embed "SourceCodePro-Regular.ttf"
};
}  // namespace

namespace caliper_web {

EmbeddedFont ui_font() { return {kUi, sizeof kUi}; }
EmbeddedFont mono_font() { return {kMono, sizeof kMono}; }

}  // namespace caliper_web
