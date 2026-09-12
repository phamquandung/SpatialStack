// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from vm_vln_msgs:msg/VlnHybridOutput.idl
// generated code does not contain a copyright notice

#ifndef VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_H_
#define VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'TRAJECTORY'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__TRAJECTORY = 0
};

/// Constant 'DISCRETE_ACTION'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__DISCRETE_ACTION = 1
};

/// Constant 'STOP'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__STOP = 0
};

/// Constant 'MOVE_FORWARD'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__MOVE_FORWARD = 1
};

/// Constant 'TURN_LEFT'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__TURN_LEFT = 2
};

/// Constant 'TURN_RIGHT'.
enum
{
  vm_vln_msgs__msg__VlnHybridOutput__TURN_RIGHT = 3
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'trajectory'
#include "nav_msgs/msg/detail/path__struct.h"
// Member 'action_codes'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/VlnHybridOutput in the package vm_vln_msgs.
typedef struct vm_vln_msgs__msg__VlnHybridOutput
{
  std_msgs__msg__Header header;
  uint32_t step_index;
  uint8_t type_of_output;
  /// When mode == trajectory → use this field
  nav_msgs__msg__Path trajectory;
  /// When mode == discrete_action → use this field
  rosidl_runtime_c__int32__Sequence action_codes;
} vm_vln_msgs__msg__VlnHybridOutput;

// Struct for a sequence of vm_vln_msgs__msg__VlnHybridOutput.
typedef struct vm_vln_msgs__msg__VlnHybridOutput__Sequence
{
  vm_vln_msgs__msg__VlnHybridOutput * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} vm_vln_msgs__msg__VlnHybridOutput__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // VM_VLN_MSGS__MSG__DETAIL__VLN_HYBRID_OUTPUT__STRUCT_H_
