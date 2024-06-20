#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sys/time.h>

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <mgard/compress_x.hpp>
#include <mpi.h>

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
#ifdef KOKKOS_ENABLE_HIP
#define MemSpace Kokkos::Experimental::HIPSpace
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
#define MemSpace Kokkos::OpenMPTargetSpace
#endif

#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>;

int rank, nproc;
MPI_Comm comm;
std::string xml_filename;      // = "adios2.xml"
std::string in_filename;       // = "xgc.f0.in.bp";
std::string out_filename;      // = "xgc.f0.out.bp";
std::string compress = "none"; // == mgard, blosc, none

typedef std::chrono::duration<double> Seconds;
typedef std::chrono::time_point<std::chrono::steady_clock,
                                std::chrono::duration<double, std::chrono::steady_clock::period>>
    TimePoint;

inline TimePoint Now() { return std::chrono::steady_clock::now(); }

std::string DimsToString(const adios2::Dims &dimensions)
{
    std::string dimensionsString("Dims(" + std::to_string(dimensions.size()) + "):[");

    for (const auto dimension : dimensions)
    {
        dimensionsString += std::to_string(dimension) + ", ";
    }
    dimensionsString.pop_back();
    dimensionsString.pop_back();
    dimensionsString += "]";
    return dimensionsString;
}

struct Timers
{
    double tread;
    double tdecomp;
    double twrite;
    double tcomp;
    Timers() : tread(0.0), tdecomp(0.0), twrite(0.0), tcomp(0.0){};
};

Timers runtest()
{
    adios2::Variable<double> var_ef_in, var_if_in;
    adios2::Variable<double> var_ef_out, var_if_out;
    adios2::ADIOS ad(xml_filename, comm);
    adios2::IO reader_io = ad.DeclareIO("ReadIO");
    adios2::IO writer_io = ad.DeclareIO("WriteIO");
    adios2::Engine reader = reader_io.Open(in_filename, adios2::Mode::ReadRandomAccess, comm);
    adios2::Engine writer = writer_io.Open(out_filename, adios2::Mode::Write, comm);
    bool decompress = false;

    var_ef_in = reader_io.InquireVariable<double>("e_f");
    // var_if_in = reader_io.InquireVariable<double>("i_f");
    auto shape = var_ef_in.Shape();

    // 4D array nplanes * 39 * nnode * 39, decomposed on dim 'nnode'
    if (shape.size() == 2)
    {
        decompress = true;
    }

    size_t nplanes = shape[0];
    size_t myplane = static_cast<size_t>(rank) % nplanes;      // 0, 1, ... nplanes-1
    size_t plane_rank = static_cast<size_t>(rank) / nplanes;   // 0, 1, ... n_per_plane
    size_t n_per_plane = static_cast<size_t>(nproc) / nplanes; // num proc per plane
    if (!n_per_plane)
    {
        n_per_plane = 1;
    }

    if (!rank)
    {
        std::cout << DimsToString(shape) << "   nplanes = " << nplanes << " decompress = " << decompress
                  << std::endl;
    }

    size_t nnode;
    size_t local_nnode;
    size_t start_nnode;
    size_t N = 0;
    if (decompress)
    {
    }
    else
    {
        nnode = shape[2];
        local_nnode = nnode / n_per_plane;
        start_nnode = local_nnode * plane_rank;

        N = shape[1] * local_nnode * shape[3]; // number of elements per vector

        std::cout << "Rank " << rank << " reads 1x" << shape[1] << "x" << local_nnode << "x" << shape[3]
                  << " array from offset " << myplane << ":0:" << start_nnode << ":0"
                  << "  nelems = " << N << std::endl;

        var_ef_in.SetSelection({{myplane, 0, start_nnode, 0}, {1, shape[1], local_nnode, shape[3]}});
        // var_if_in.SetSelection({{myplane, 0, start_nnode, 0}, {1, shape[1], local_nnode, shape[3]}});
    }

    // Allocate vectors on device.
    typedef Kokkos::View<double *, Kokkos::LayoutLeft, MemSpace> ViewVectorType;
    ViewVectorType e_f("e_f", N);
    // ViewVectorType i_f("i_f", N);

    TimePoint tstart_read = Now();

    reader.Get<double>(var_ef_in, e_f);
    // reader.Get<double>(var_if_in, i_f);
    reader.PerformGets();
    TimePoint tend_read = Now();
    reader.Close();

    TimePoint tstart_decomp = Now();
    if (decompress)
    {
    }
    TimePoint tend_decomp = Now();

    if (compress == "mgard")
    {
        // ViewVectorType e_f_comp("e_f", N);
    }

    TimePoint tstart_comp = Now();
    double *e_f_out; // could be a pointer on device or host
    if (compress == "none")
    {
        var_ef_out = writer_io.DefineVariable<double>(
            "e_f", {nplanes, shape[1], local_nnode * n_per_plane, shape[3]}, {myplane, 0, start_nnode, 0},
            {1, shape[1], local_nnode, shape[3]});
        /*var_if_out = writer_io.DefineVariable<double>(
            "i_f", {nplanes, shape[1], local_nnode * n_per_plane, shape[3]}, {myplane, 0, start_nnode, 0},
            {1, shape[1], local_nnode, shape[3]});*/
        e_f_out = e_f.data(); // on device
    }
    else
    {
        mgard_x::data_type mgardType = mgard_x::data_type::Double;
        mgard_x::DIM mgardDim = 3;
        std::vector<mgard_x::SIZE> mgardCount = {shape[1], local_nnode, shape[3]};
        bool hasTolerance = false;
        double tolerance = 0.001;
        double s = 0.0;
        auto errorBoundType = mgard_x::error_bound_type::REL;
        e_f_out = (double *)malloc(N * sizeof(double)); // on host
                                                        // void *compressedData = (void *)e_f_out;
        void *compressedData = (void *)(e_f_out);
        size_t compressedSize = N;

        mgard_x::compress(mgardDim, mgardType, mgardCount, tolerance, s, errorBoundType, e_f.data(),
                          compressedData, compressedSize, true);

        var_ef_out = writer_io.DefineVariable<double>("e_f", {nplanes, compressedSize}, {myplane, 0},
                                                      {1, compressedSize});
    }
    TimePoint tend_comp = Now();

    std::cout << "Rank " << rank << " write block " << DimsToString(var_ef_out.Count())
              << " compress = " << compress << std::endl;
    TimePoint tstart_write = Now();
    writer.BeginStep();
    writer.Put(var_ef_out, e_f_out);
    // writer.Put(var_if_out, i_f);
    writer.EndStep();
    writer.Close();
    TimePoint tend_write = Now();

    Timers t;
    t.tread = Seconds(tend_read - tstart_read).count();
    t.tdecomp = Seconds(tend_decomp - tstart_decomp).count();
    t.tread = Seconds(tend_read - tstart_read).count();
    t.tcomp = Seconds(tend_comp - tstart_comp).count();

    return t;
}

void print_timers(Timers &timers)
{
    std::vector<Timers> tv(nproc);
    MPI_Gather(&timers, sizeof(Timers), MPI_CHAR, tv.data(), sizeof(Timers), MPI_CHAR, 0, comm);

    if (!rank)
    {
        Timers maxs;
        std::cout << "Rank :      read   decompression  write  compression (seconds)\n";
        std::cout << "--------------------------------------------------------------\n";
        for (int r = 0; r < nproc; ++r)
        {
            std::cout << std::setw(5) << r << ":   " << std::fixed << std::setw(8) << std::setprecision(3)
                      << tv[r].tread << "   " << std::setw(8) << std::setprecision(3) << tv[r].tdecomp
                      << "   " << std::setw(8) << std::setprecision(3) << tv[r].twrite << "   "
                      << std::setw(8) << std::setprecision(3) << tv[r].tcomp << "\n";
            maxs.tread = (maxs.tread > tv[r].tread ? maxs.tread : tv[r].tread);
            maxs.twrite = (maxs.twrite > tv[r].twrite ? maxs.twrite : tv[r].twrite);
            maxs.tcomp = (maxs.tcomp > tv[r].tcomp ? maxs.tcomp : tv[r].tcomp);
            maxs.tdecomp = (maxs.tdecomp > tv[r].tdecomp ? maxs.tdecomp : tv[r].tdecomp);
        }
        std::cout << "  max:   " << std::fixed << std::setw(8) << std::setprecision(3) << maxs.tread << "   "
                  << std::setw(8) << std::setprecision(3) << maxs.tdecomp << "   " << std::setw(8)
                  << std::setprecision(3) << maxs.twrite << "   " << std::setw(8) << std::setprecision(3)
                  << maxs.tcomp << "\n";
        std::cout << std::endl;
    }
}

void Usage(const char *prgname)
{
    std::cout << "Usage: " << prgname << " XML-file  f0_input  f0_output none|mgard" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        Usage(argv[0]);
        return 1;
    }

    xml_filename = argv[1];
    in_filename = argv[2];
    out_filename = argv[3];
    compress = argv[4];

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    if (!rank)
    {
        std::cout << "Test f0 data\n  input    = " << in_filename << "\n  output   = " << out_filename
                  << "\n  settings = " << xml_filename << "\n  memspace = " << MemSpace::name() << std::endl;
    }

    Kokkos::initialize(argc, argv);
    {
        Timers t = runtest();
        print_timers(t);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
