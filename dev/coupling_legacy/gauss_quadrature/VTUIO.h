/// @date 2024-08-08
/// @file VtuIO.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <array>
#include "pugixml.hpp"
#include <vector>
#include <sstream>

namespace {
[[maybe_unused]] void processStringStream(std::stringstream& sstream, const double& x) { sstream << x << " "; }

template <size_t dim, typename T>
void processStringStream(std::stringstream& sstream, const std::array<T, dim>& x) {
    for (size_t i = 0; i < dim; i++) {
        sstream << x[i] << " ";
    }
}

template <typename T>
struct value_size_of_vector;

template <typename T>
struct value_size_of_vector<std::vector<T>> {
    static constexpr std::size_t size = 1;
};

template <typename T, std::size_t N>
struct value_size_of_vector<std::vector<std::array<T, N>>> {
    static constexpr std::size_t size = N;
};


// 将一个vector转换成字符串
template <typename T, template <typename> typename Vector>
Vector<std::string> vectors_to_strings(const Vector<T>& first) {
    Vector<std::string> result;
    std::stringstream   sstream;
    for (auto i : first)
        processStringStream(sstream, i);
    result.push_back(sstream.str());
    return result;
}

template <typename T, template <typename> typename Vector, typename... Args>
Vector<std::string> vectors_to_strings(const Vector<T>& first, const Args&... rest) {
    const Vector<std::string>& temp1 = vectors_to_strings(first);
    const Vector<std::string>& temp2 = vectors_to_strings(rest...);
    Vector<std::string>        temp;
    for (auto i : temp1)
        temp.push_back(i);
    for (auto i : temp2)
        temp.push_back(i);
    return temp;
}
} // namespace

namespace IO {
template <template <typename> typename Vector>
void write_particles_to_vtu(
    const int& length, 
    const std::string& points, 
    const std::string& filename,
    const Vector<std::string>& contents, 
    const Vector<std::string>& tags,
    const Vector<size_t>& value_sizes) 
{
    const size_t num = contents.size();
    // ASSERT(num == tags.size() && num == value_sizes.size());
    pugi::xml_document xml_node;
    auto               vtkfile_node          = xml_node.append_child("VTKFile");
    vtkfile_node.append_attribute("type")    = "UnstructuredGrid";
    vtkfile_node.append_attribute("version") = "0.1";
    {
        auto u_node = vtkfile_node.append_child("UnstructuredGrid");
        {
            auto piece                               = u_node.append_child("Piece");
            piece.append_attribute("NumberOfPoints") = std::to_string(length).c_str();
            piece.append_attribute("NumberOfCells")  = "0";
            {
                // points coordinates
                auto node_1     = piece.append_child("Points");
                auto data_array = node_1.append_child("DataArray");

                data_array.append_attribute("type")               = "Float64";
                data_array.append_attribute("NumberOfComponents") = "3";
                data_array.append_attribute("format")             = "ascii";
                data_array.append_child(pugi::node_pcdata).set_value(points.c_str());
            }
            {
                // cells
                auto cells_node                        = piece.append_child("Cells");
                auto dataarray_1                       = cells_node.append_child("DataArray");
                dataarray_1.append_attribute("type")   = "UInt32";
                dataarray_1.append_attribute("Name")   = "connectivity";
                dataarray_1.append_attribute("format") = "ascii";
                dataarray_1.append_child(pugi::node_pcdata).set_value("");
                auto dataarray_2                       = cells_node.append_child("DataArray");
                dataarray_2.append_attribute("type")   = "UInt32";
                dataarray_2.append_attribute("Name")   = "offsets";
                dataarray_2.append_attribute("format") = "ascii";
                dataarray_2.append_child(pugi::node_pcdata).set_value("");
                auto dataarray_3                       = cells_node.append_child("DataArray");
                dataarray_3.append_attribute("type")   = "UInt8";
                dataarray_3.append_attribute("Name")   = "types";
                dataarray_3.append_attribute("format") = "ascii";
                dataarray_3.append_child(pugi::node_pcdata).set_value("");
            }

            auto point_data = piece.append_child("PointData");
            // point_data.append_attribute("Vectors") = tags[0].c_str();
            for (size_t i = 0; i < static_cast<size_t>(num); i++) {
                // points data
                // spdlog::info(" {}: {}", tags[i], contents[i]);
                auto data_array = point_data.append_child("DataArray");

                data_array.append_attribute("type")               = "Float64";
                data_array.append_attribute("Name")               = tags[i].c_str();
                data_array.append_attribute("NumberOfComponents") = std::to_string(value_sizes[i]).c_str();
                data_array.append_attribute("format")             = "ascii";
                data_array.append_child(pugi::node_pcdata).set_value(contents[i].c_str());
            }
        }
    }
    xml_node.save_file(filename.c_str());
}

// 文件名、点坐标、标签、内容
template <typename T, template <typename> typename Vector, typename... Args>
void write_particles_to_vtu(
    const std::string& filename, 
    const Vector<T>& points_raw, 
    const Vector<std::string>& tags,
    const Args&... contents_raw) 
{
    // ASSERT_INFO(sizeof...(Args) == tags.size(), "{} == {}", sizeof...(Args), tags.size());
    if constexpr (sizeof...(Args)>0) {
        auto points   = vectors_to_strings(points_raw)[0];
        auto contents = vectors_to_strings(contents_raw...);
        std::vector<size_t> value_sizes = {value_size_of_vector<Args>::size...};
        write_particles_to_vtu(points_raw.size(), points, filename, contents, tags, value_sizes);
    } else {
        // ASSERT_INFO(sizeof...(Args) > 0, "No data to write");
    }
}
}
