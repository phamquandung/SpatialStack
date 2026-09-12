// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#ifndef VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_HPP_
#define VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'trajectory'
#include "nav_msgs/msg/detail/path__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__vm_vln_msgs__msg__VlnHybridOutput __attribute__((deprecated))
#else
# define DEPRECATED__vm_vln_msgs__msg__VlnHybridOutput __declspec(deprecated)
#endif

namespace vm_vln_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct VlnHybridOutput_
{
  using Type = VlnHybridOutput_<ContainerAllocator>;

  explicit VlnHybridOutput_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    trajectory(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->step_index = 0ul;
      this->type_of_output = 0;
    }
  }

  explicit VlnHybridOutput_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    trajectory(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->step_index = 0ul;
      this->type_of_output = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _step_index_type =
    uint32_t;
  _step_index_type step_index;
  using _type_of_output_type =
    uint8_t;
  _type_of_output_type type_of_output;
  using _trajectory_type =
    nav_msgs::msg::Path_<ContainerAllocator>;
  _trajectory_type trajectory;
  using _action_codes_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _action_codes_type action_codes;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__step_index(
    const uint32_t & _arg)
  {
    this->step_index = _arg;
    return *this;
  }
  Type & set__type_of_output(
    const uint8_t & _arg)
  {
    this->type_of_output = _arg;
    return *this;
  }
  Type & set__trajectory(
    const nav_msgs::msg::Path_<ContainerAllocator> & _arg)
  {
    this->trajectory = _arg;
    return *this;
  }
  Type & set__action_codes(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->action_codes = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t TRAJECTORY =
    0u;
  static constexpr uint8_t DISCRETE_ACTION =
    1u;
  static constexpr uint8_t STOP =
    0u;
  static constexpr uint8_t MOVE_FORWARD =
    1u;
  static constexpr uint8_t TURN_LEFT =
    2u;
  static constexpr uint8_t TURN_RIGHT =
    3u;

  // pointer types
  using RawPtr =
    vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> *;
  using ConstRawPtr =
    const vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__vm_vln_msgs__msg__VlnHybridOutput
    std::shared_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__vm_vln_msgs__msg__VlnHybridOutput
    std::shared_ptr<vm_vln_msgs::msg::VlnHybridOutput_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const VlnHybridOutput_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->step_index != other.step_index) {
      return false;
    }
    if (this->type_of_output != other.type_of_output) {
      return false;
    }
    if (this->trajectory != other.trajectory) {
      return false;
    }
    if (this->action_codes != other.action_codes) {
      return false;
    }
    return true;
  }
  bool operator!=(const VlnHybridOutput_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct VlnHybridOutput_

// alias to use template instance with default allocator
using VlnHybridOutput =
  vm_vln_msgs::msg::VlnHybridOutput_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::TRAJECTORY;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::DISCRETE_ACTION;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::STOP;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::MOVE_FORWARD;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::TURN_LEFT;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t VlnHybridOutput_<ContainerAllocator>::TURN_RIGHT;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace vm_vln_msgs

#endif  // VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_HPP_
