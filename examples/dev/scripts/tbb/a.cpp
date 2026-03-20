#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>

template <typename T> T SerialSumFoo(const T *a, size_t n) {
    T sum = 0;
    for (size_t i = 0; i != n; ++i)
        sum += a[i];
    return sum;
}

// y = ax
template <typename T> void add(T *y, const T *x, T alpha, size_t n) {
    for (size_t i = 0; i != n; ++i)
        y[i] += alpha * x[i];
}


using namespace oneapi::tbb;

template <typename T> class Add {
    const T *_x = nullptr;
    T *_y = nullptr;
    T _alpha = 1.0f;

  public:
    void operator()(const oneapi::tbb::blocked_range<size_t> &r) const {
        const T *x = _x;
        T *y = _y;
        T alpha = _alpha;
        for (size_t i = r.begin(); i != r.end(); ++i)
            y[i] += alpha*x[i];
    }
    // ApplyFoo(float a[]) : my_a(a) {}
};

int main() {
    oneapi::tbb::global_control ctrl(oneapi::tbb::global_control::max_allowed_parallelism, 4);
    oneapi::tbb::parallel_for(0, 10, [](int i) { /* ... */ });

    std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> b{1, 3, 5, 7, 9, 7, 5, 3, 1};
    auto alpha = SerialSumFoo(a.data(), a.size());

    add(a.data(), b.data(), alpha, a.size());
    printf("%f\n", alpha);
    return 0;
}