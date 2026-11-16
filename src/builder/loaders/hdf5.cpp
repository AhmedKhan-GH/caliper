// The HDF5 reader. This build has no libhdf5 (the build never fetches or
// installs one), so the reader is a refusal that points to the safe routes:
// a conversion to safetensors or .npy, or the Python loader, which reads the
// file with h5py. docs/plans/loaders.md ("HDF5") records the plan.
#include <caliper_builder/loaders/tensor_readers.h>

namespace caliper_builder {

bool hdf5_available() { return false; }

TensorReadResult read_hdf5(const std::string& path, const std::string&, const TensorFilter&, const ReaderLimits&) {
    TensorReadResult result;
    result.error = Error("loader.hdf5_not_available",
                         "This build of Caliper has no HDF5 library, so it cannot read HDF5 files; convert the file "
                         "to safetensors or .npy, or use the Python loader, which reads it with h5py.",
                         path);
    return result;
}

}  // namespace caliper_builder
