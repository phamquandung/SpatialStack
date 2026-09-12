// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#ifndef VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__BUILDER_HPP_
#define VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "vm_vln_msgs/msg/detail/vln_hybrid_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace vm_vln_msgs
{

namespace msg
{

namespace builder
{

class Init_VlnHybridOutput_action_codes
{
public:
  explicit Init_VlnHybridOutput_action_codes(::vm_vln_msgs::msg::VlnHybridOutput & msg)
  : msg_(msg)
  {}
  ::vm_vln_msgs::msg::VlnHybridOutput action_codes(::vm_vln_msgs::msg::VlnHybridOutput::_action_codes_type arg)
  {
    msg_.action_codes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::vm_vln_msgs::msg::VlnHybridOutput msg_;
};

class Init_VlnHybridOutput_trajectory
{
public:
  explicit Init_VlnHybridOutput_trajectory(::vm_vln_msgs::msg::VlnHybridOutput & msg)
  : msg_(msg)
  {}
  Init_VlnHybridOutput_action_codes trajectory(::vm_vln_msgs::msg::VlnHybridOutput::_trajectory_type arg)
  {
    msg_.trajectory = std::move(arg);
    return Init_VlnHybridOutput_action_codes(msg_);
  }

private:
  ::vm_vln_msgs::msg::VlnHybridOutput msg_;
};

class Init_VlnHybridOutput_type_of_output
{
public:
  explicit Init_VlnHybridOutput_type_of_output(::vm_vln_msgs::msg::VlnHybridOutput & msg)
  : msg_(msg)
  {}
  Init_VlnHybridOutput_trajectory type_of_output(::vm_vln_msgs::msg::VlnHybridOutput::_type_of_output_type arg)
  {
    msg_.type_of_output = std::move(arg);
    return Init_VlnHybridOutput_trajectory(msg_);
  }

private:
  ::vm_vln_msgs::msg::VlnHybridOutput msg_;
};

class Init_VlnHybridOutput_step_index
{
public:
  explicit Init_VlnHybridOutput_step_index(::vm_vln_msgs::msg::VlnHybridOutput & msg)
  : msg_(msg)
  {}
  Init_VlnHybridOutput_type_of_output step_index(::vm_vln_msgs::msg::VlnHybridOutput::_step_index_type arg)
  {
    msg_.step_index = std::move(arg);
    return Init_VlnHybridOutput_type_of_output(msg_);
  }

private:
  ::vm_vln_msgs::msg::VlnHybridOutput msg_;
};

class Init_VlnHybridOutput_header
{
public:
  Init_VlnHybridOutput_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VlnHybridOutput_step_index header(::vm_vln_msgs::msg::VlnHybridOutput::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_VlnHybridOutput_step_index(msg_);
  }

private:
  ::vm_vln_msgs::msg::VlnHybridOutput msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::vm_vln_msgs::msg::VlnHybridOutput>()
{
  return vm_vln_msgs::msg::builder::Init_VlnHybridOutput_header();
}

}  // namespace vm_vln_msgs

#endif  // VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__BUILDER_HPP_
