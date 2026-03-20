#include "common.h"
template <typename GridState>
struct Grid
{
    static constexpr std::size_t _dim = GridState::_dim;

    using value_type = typename GridState::value_type;
    using state_type = GridState;
    using index_type = Index<_dim>;

    index_type grid_size;

    Grid(const index_type & _grid_size) : grid_size(_grid_size)
    {
        if constexpr (_dim == 1)
        {
            states.resize(grid_size.i);
        }
        if constexpr (_dim == 2)
        {
            states.resize(grid_size.i * grid_size.j);
        }
        if constexpr (_dim == 3)
        {
            states.resize(grid_size.i * grid_size.j * grid_size.k);
        }
    }

    size_t size() const
    {
        return states.size();
    }

    void copy_from(const std::vector<state_type> &data)
    {
        states = data;
    }

    void copy_to(std::vector<state_type> &data) const
    {
        data = states;
    }

    template <typename Function>
    void set_state(const Function function)
    {
        if constexpr (_dim == 2)
        {
            for (size_t i = 0; i < grid_size.i; i++)
            {
                for (size_t j = 0; j < grid_size.j; j++)
                {
                    auto& state = get_state({i, j});
                    function(state,{i, j});
                }
            }
        }
        if constexpr (_dim == 3)
        {
            for (size_t i = 0; i < grid_size.i; i++)
            {
                for (size_t j = 0; j < grid_size.j; j++)
                {
                    for (size_t k = 0; k < grid_size.k; k++)
                    {
                        auto& state = get_state({i, j, k});
                        function(state,{i, j, k});
                    }
                }
            }
        }
    }

    size_t calculate_hash(const index_type offset) const
    {
        if constexpr (_dim == 1)
            return offset.i;
        else if constexpr (_dim == 2)
            return offset.j * grid_size.i + offset.i;
        else if constexpr (_dim == 3)
            return offset.k * grid_size.i * grid_size.j + offset.j * grid_size.i + offset.i;
    }

    state_type &get_state(index_type offset)
    {
        return states[calculate_hash(offset)];
    }

// private:
public:
    std::vector<state_type> states;
};