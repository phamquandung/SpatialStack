// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#ifndef VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__TRAITS_HPP_
#define VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "vm_vln_msgs/msg/detail/vln_hybrid_output__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'trajectory'
#include "nav_msgs/msg/detail/path__traits.hpp"

namespace vm_vln_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const VlnHybridOutput & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: step_index
  {
    out << "step_index: ";
    rosidl_generator_traits::value_to_yaml(msg.step_index, out);
    out << ", ";
  }

  // member: type_of_output
  {
    out << "type_of_output: ";
    rosidl_generator_traits::value_to_yaml(msg.type_of_output, out);
    out << ", ";
  }

  // member: trajectory
  {
    out << "trajectory: ";
    to_flow_style_yaml(msg.trajectory, out);
    out << ", ";
  }

  // member: action_codes
  {
    if (msg.action_codes.size() == 0) {
      out << "action_codes: []";
    } else {
      out << "action_codes: [";
      size_t pending_items = msg.action_codes.size();
      for (auto item : msg.action_codes) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const VlnHybridOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: step_index
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "step_index: ";
    rosidl_generator_traits::value_to_yaml(msg.step_index, out);
    out << "\n";
  }

  // member: type_of_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "type_of_output: ";
    rosidl_generator_traits::value_to_yaml(msg.type_of_output, out);
    out << "\n";
  }

  // member: trajectory
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "trajectory:\n";
    to_block_style_yaml(msg.trajectory, out, indentation + 2);
  }

  // member: action_codes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.action_codes.size() == 0) {
      out << "action_codes: []\n";
    } else {
      out << "action_codes:\n";
      for (auto item : msg.action_codes) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const VlnHybridOutput & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace vm_vln_msgs

namespace rosidl_generator_traits
{

[[deprecated("use vm_vln_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const vm_vln_msgs::msg::VlnHybridOutput & msg,
  std::ostream & out, size_t indentation = 0)
{
  vm_vln_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use vm_vln_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const vm_vln_msgs::msg::VlnHybridOutput & msg)
{
  return vm_vln_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<vm_vln_msgs::msg::VlnHybridOutput>()
{
  return "vm_vln_msgs::msg::VlnHybridOutput";
}

template<>
inline const char * name<vm_vln_msgs::msg::VlnHybridOutput>()
{
  return "vm_vln_msgs/msg/VlnHybridOutput";
}

template<>
struct has_fixed_size<vm_vln_msgs::msg::VlnHybridOutput>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<vm_vln_msgs::msg::VlnHybridOutput>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<vm_vln_msgs::msg::VlnHybridOutput>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__TRAITS_HPP_
