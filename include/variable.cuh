#ifndef VARIABLE_CUH
#define VARIABLE_CUH

#include "../include/rand.cuh"
#include "../include/utils.cuh"
#include "../include/shared_ptr.cuh"
#include "../include/smart_object.cuh"

#include <vector>
#include <fstream>

class Variable
{
public:
    inline static std::vector<natural> sizes;
    inline static dev_shared_ptr<RandState> dev_rand_states;
    dev_shared_ptr<real> dev_data;
    dev_shared_ptr<real> dev_grad;
    natural size, rows, cols;

    Variable(const natural size_, const bool requires_grad = true, const bool rand = false, const natural rows_ = 0, const natural cols_ = 0);
    Variable() = default;
    void print(const std::string &what, natural col) const;
    void save(const std::string &file_name, const std::string &what, natural col) const;
    void zero(smart_stream stream) const;
    void zero_grad(smart_stream stream) const;
    void glorot() const;
    void set_value(const real value, smart_stream stream) const;
    static void initialize_random();
};

#endif